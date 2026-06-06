"""
mujoco_runner.py

Runs a dual-arm box transfer task in MuJoCo using the Aloha robot model.
Loads the scene, computes a handoff zone, then executes a sequence of phases
where each phase drives one arm via IK and rate-limited joint control.

Per-phase reward, is_done, and gripper functions are provided by
reward_function.py. Phases that time out are optionally decomposed into
sub-phases. Reward history is plotted and saved to outputs/reward_curve.png
on exit.

Usage:
    python src/mujoco_runner.py
    python src/mujoco_runner.py --task "move box from table B to table A"
"""

import sys
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

from mujoco_playground._src.manipulation.aloha import base as aloha_base

from types_ import Phase, TaskDecomposition
from motion_descriptor import parse_task
from reward_function import build_phase_functions,PhaseFunctions, decompose_phase, GRIP_OPEN, GRIP_CLOSED
from handoff_resolver import compute_handoff_zone


MENAGERIE_PATH = Path(__file__).parent.parent / "aloha_menagerie"
SCENE_XML      = Path(__file__).parent.parent / "assets" / "scene_dual_arm.xml"
OUTPUTS_DIR    = Path(__file__).parent.parent / "outputs"

JOINT_SLICE_A = slice(0, 7)
JOINT_SLICE_B = slice(7, 14)

GRIP_IDX_A = 6
GRIP_IDX_B = 13

# small z offset applied to box reach/grasp IK targets so the orientation
# constraint can be satisfied without the EE colliding with the table surface
BOX_REACH_Z_OFFSET = 0.02

WELD_DISTANCE_THRESHOLD = 0.1

MIN_EE_Z = 0.05

MAX_STEPS_PER_PHASE    = 1500
MAX_DECOMPOSE_ATTEMPTS = 1
MAX_CTRL_DELTA         = 0.008


# -------------------------
# MODEL LOADING
# -------------------------

def load_model():
    """Loads the MuJoCo model from XML with Aloha assets injected."""
    assets = aloha_base.get_assets()
    aloha_assets_dir = MENAGERIE_PATH / "aloha" / "assets"

    if aloha_assets_dir.exists():
        for f in aloha_assets_dir.iterdir():
            if f.is_file():
                assets[f.name] = f.read_bytes()

    xml_str = SCENE_XML.read_text()
    return mujoco.MjModel.from_xml_string(xml_str, assets)


# -------------------------
# SCENE INFO
# -------------------------

class SceneInfo:
    """Caches MuJoCo IDs and fixed scene geometry used throughout the runner."""

    def __init__(self, model):
        self.box_body      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.left_gripper  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
        self.right_gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")

        self.box_qpos_adr = None
        for i in range(model.njnt):
            if model.jnt_bodyid[i] == self.box_body:
                self.box_qpos_adr = model.jnt_qposadr[i]
                break
        if self.box_qpos_adr is None:
            raise RuntimeError("Could not find free joint for box body")

        self.table_a_x  = -0.96
        self.table_b_x  =  0.96
        self.box_rest_z =  0.035
        self.handoff_z  =  0.035
        self.handoff_x  =  0.0


# -------------------------
# UTIL
# -------------------------

def zone_pos(scene, zone):
    """Returns the world position of a named zone (A, B, or center)."""
    if zone == "A":
        return np.array([scene.table_a_x, 0.0, scene.box_rest_z])
    if zone == "B":
        return np.array([scene.table_b_x, 0.0, scene.box_rest_z])
    if zone in ("center", "handoff_zone"):
        return np.array([scene.handoff_x, 0.0, scene.handoff_z])
    raise ValueError(f"Unknown zone: {zone}")


def set_box_pose(model, data, scene, pos):
    """Teleports the box to pos and zeroes its velocity."""
    adr = scene.box_qpos_adr
    data.qpos[adr:adr+3]   = pos
    data.qpos[adr+3]        = 1.0
    data.qpos[adr+4:adr+7] = 0.0
    data.qvel[:]            = 0
    mujoco.mj_forward(model, data)


def get_box_pos(data, scene):
    """Returns the current box position from qpos."""
    adr = scene.box_qpos_adr
    return data.qpos[adr:adr+3].copy()


def enable_weld(model, data, weld_id):
    """Activates a weld equality constraint in both model defaults and current state."""
    model.eq_active0[weld_id] = 1
    data.eq_active[weld_id]   = 1
    mujoco.mj_forward(model, data)


def disable_weld(model, data, weld_id):
    """Deactivates a weld equality constraint in both model defaults and current state."""
    model.eq_active0[weld_id] = 0
    data.eq_active[weld_id]   = 0
    mujoco.mj_forward(model, data)


def get_weld_id(model, name):
    """Returns the equality constraint index for the named weld, or -1 if not found."""
    for i in range(model.neq):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) == name:
            return i
    return -1


def build_scene_state(data, scene, handoff_pos, goal_pos):
    """
    Builds the scene state dict passed to reward_function at each phase entry.
    Includes live positions, EE rotation matrices, and the box z offset used
    by both IK and reward targets so they stay consistent.
    """
    return {
        "box_pos":           get_box_pos(data, scene),
        "handoff_pos":       handoff_pos,
        "goal_pos":          goal_pos,
        "ee_a":              data.site_xpos[scene.left_gripper].copy(),
        "ee_b":              data.site_xpos[scene.right_gripper].copy(),
        "ee_a_xmat":         data.site_xmat[scene.left_gripper].reshape(3, 3).copy(),
        "ee_b_xmat":         data.site_xmat[scene.right_gripper].reshape(3, 3).copy(),
        "min_ee_z":          MIN_EE_Z,
        "box_reach_z_offset": BOX_REACH_Z_OFFSET,
    }


# -------------------------
# IK SOLVER
# -------------------------

# minimum z any part of the arm should reach during motion
TRAJ_MIN_Z        = 0.05
# clearance height the EE must pass through before descending to target
TRAJ_WAYPOINT_Z   = 0.25
# robot base radius — EE must stay outside this in XY when near base height
ROBOT_BASE_RADIUS = 0.12


def _base_pos(arm):
    """Returns the XY base position for the given arm."""
    return np.array([-0.469, -0.019]) if arm == "A" else np.array([0.469, -0.019])


def solve_ik(model, data, joint_slice, site_id, ee_target,
             enforce_orientation=False, q_seed=None, arm=None):
    """
    Solves IK using SLSQP with explicit constraints.

    Constraints enforced:
      - EE z must stay above TRAJ_MIN_Z (no going through table)
      - EE must stay outside ROBOT_BASE_RADIUS in XY when near base height
      - approach axis horizontal and Z axis up when enforce_orientation is True

    Returns (joint_config, position_error).
    Restores sim state after solving.
    """
    lo       = model.jnt_range[joint_slice, 0]
    hi       = model.jnt_range[joint_slice, 1]
    q0       = q_seed if q_seed is not None else data.qpos[joint_slice].copy()
    q_backup = data.qpos.copy()
    v_backup = data.qvel.copy()

    def get_ee(q):
        data.qpos[joint_slice] = q
        mujoco.mj_forward(model, data)
        return data.site_xpos[site_id].copy()

    def objective(q):
        ee = get_ee(q)
        pos_err = np.linalg.norm(ee - ee_target)

        if not enforce_orientation:
            return float(pos_err)

        R                 = data.site_xmat[site_id].reshape(3, 3)
        approach_tilt_err = abs(float(R[:, 0][2]))
        up_err            = 1.0 - float(np.dot(R[:, 2], np.array([0.0, 0.0, 1.0])))

        to_target      = ee_target - ee
        to_target_h    = to_target.copy()
        to_target_h[2] = 0.0
        norm           = np.linalg.norm(to_target_h)
        if norm > 1e-6:
            to_target_h /= norm
            approach_h      = R[:, 0].copy()
            approach_h[2]   = 0.0
            ah_norm         = np.linalg.norm(approach_h)
            if ah_norm > 1e-6:
                approach_h /= ah_norm
            approach_dir_err = 1.0 - float(np.dot(approach_h, to_target_h))
        else:
            approach_dir_err = 0.0

        return float(pos_err
                     + 0.5 * up_err
                     + 0.5 * approach_tilt_err
                     + 0.3 * approach_dir_err)

    constraints = []

    # EE must stay above table surface
    constraints.append({
        "type": "ineq",
        "fun": lambda q: float(get_ee(q)[2] - TRAJ_MIN_Z),
    })

    # EE must stay outside robot base footprint in XY when below arm height
    if arm is not None:
        base_xy = _base_pos(arm)
        constraints.append({
            "type": "ineq",
            "fun": lambda q: float(
                np.linalg.norm(get_ee(q)[:2] - base_xy) - ROBOT_BASE_RADIUS
                if get_ee(q)[2] < 0.20 else 1.0
            ),
        })

    bounds = [(float(lo[i]), float(hi[i])) for i in range(len(lo))]

    res = minimize(
        objective, q0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    data.qpos[joint_slice] = res.x
    mujoco.mj_forward(model, data)
    actual_cost = float(np.linalg.norm(data.site_xpos[site_id] - ee_target))

    data.qpos[:] = q_backup
    data.qvel[:] = v_backup
    mujoco.mj_forward(model, data)

    return np.clip(res.x, lo, hi), actual_cost


def solve_ik_with_waypoints(model, data, joint_slice, site_id, ee_target,
                             enforce_orientation=False, q_seed=None, arm=None):
    """
    Solves IK in two stages using waypoints to prevent self-collision trajectories.

    Stage 1: move EE to a clearance waypoint directly above the target at
             TRAJ_WAYPOINT_Z height. This forces the arm to lift over obstacles
             and clear the robot base before committing to the final position.
    Stage 2: solve IK for the actual target, seeded from the waypoint solution.

    Returns (joint_config, position_error) for the final target.
    """
    # stage 1: lift waypoint above target
    waypoint    = ee_target.copy()
    waypoint[2] = max(ee_target[2], TRAJ_WAYPOINT_Z)

    # only solve waypoint if target is below clearance height
    if ee_target[2] < TRAJ_WAYPOINT_Z - 0.05:
        q_waypoint, wp_cost = solve_ik(
            model, data, joint_slice, site_id, waypoint,
            enforce_orientation=False,
            q_seed=q_seed,
            arm=arm,
        )
        print(f"[ik] Waypoint solved: z={waypoint[2]:.3f} cost={wp_cost:.4f}")
    else:
        q_waypoint = q_seed

    # stage 2: solve for actual target seeded from waypoint
    q_final, final_cost = solve_ik(
        model, data, joint_slice, site_id, ee_target,
        enforce_orientation=enforce_orientation,
        q_seed=q_waypoint,
        arm=arm,
    )

    return q_final, final_cost


# -------------------------
# EE TARGET FOR IK
# -------------------------

def get_ee_target(phase, scene, data, handoff_pos, goal_pos):
    """
    Returns the IK position target for the gripper site for this phase.

    Box reach/grasp targets include BOX_REACH_Z_OFFSET so the orientation
    constraint can be satisfied without the EE hitting the table. This offset
    is also passed through scene_state so reward_function uses the same value,
    keeping IK target and reward target consistent.
    """
    action = phase.action
    target = phase.target

    def clamp_z(t):
        t    = t.copy()
        t[2] = max(t[2], MIN_EE_Z)
        return t

    if action == "reach":
        if target == "box":
            t    = get_box_pos(data, scene).copy()
            t[2] = t[2] + BOX_REACH_Z_OFFSET
            return t
        if target in ("center", "handoff_zone"):
            t    = handoff_pos.copy()
            t[2] = max(t[2] + 0.05, MIN_EE_Z)
            return clamp_z(t)
        if target in ("A", "B"):
            t    = zone_pos(scene, target).copy()
            t[2] = max(t[2] + 0.05, MIN_EE_Z)
            return clamp_z(t)
        t    = handoff_pos.copy()
        t[2] = max(t[2] + 0.05, MIN_EE_Z)
        return clamp_z(t)

    if action == "place":
        if target in ("center", "handoff_zone"):
            t    = handoff_pos.copy()
            t[2] = max(t[2] + 0.05, MIN_EE_Z)
            return clamp_z(t)
        t    = goal_pos.copy()
        t[2] = max(t[2] + 0.05, MIN_EE_Z)
        return clamp_z(t)

    if action == "clear_obstacle":
        t    = data.xpos[scene.box_body].copy()
        t[2] = max(t[2] + 0.10, MIN_EE_Z)
        return clamp_z(t)

    t    = handoff_pos.copy()
    t[2] = max(t[2] + 0.05, MIN_EE_Z)
    return clamp_z(t)


# -------------------------
# MAIN RUN
# -------------------------

def run(task_str):
    """Parses the task, sets up the scene, and runs the phase execution loop."""
    print(f"\nTask: {task_str}")

    task = parse_task(task_str)
    print("\n=== TASK PHASES ===")
    for p in task.phases:
        print(f"  phase={p.phase_id} arm={p.arm} action={p.action} target={p.target}")
    print("===================\n")

    model = load_model()
    data  = mujoco.MjData(model)
    scene = SceneInfo(model)

    weld_left_id  = get_weld_id(model, "weld_left")
    weld_right_id = get_weld_id(model, "weld_right")
    weld_active   = False

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    home_qpos_a = data.qpos[JOINT_SLICE_A].copy()
    home_qpos_b = data.qpos[JOINT_SLICE_B].copy()

    start_pos = np.array([
        scene.table_a_x if task.start_zone == "A" else scene.table_b_x,
        0.0,
        scene.box_rest_z,
    ])
    set_box_pose(model, data, scene, start_pos)

    print(f"[init] box pos     : {get_box_pos(data, scene)}")
    left_gripper_init_pos = data.site_xpos[scene.left_gripper].copy()
    right_gripper_init_pos = data.site_xpos[scene.right_gripper].copy()
    print(f"[init] left EE     : {data.site_xpos[scene.left_gripper]}")
    print(f"[init] right EE    : {data.site_xpos[scene.right_gripper]}")

    site_a = scene.left_gripper
    site_b = scene.right_gripper

    handoff = compute_handoff_zone(
        model, data,
        site_a, site_b,
        JOINT_SLICE_A, JOINT_SLICE_B,
        table_z=scene.handoff_z,
        n_samples=800,
    )
    handoff_pos = handoff["center"]

    print("[init] Pre-solving center seeds...")
    center_target    = handoff_pos.copy()
    center_target[2] = max(center_target[2] + 0.05, MIN_EE_Z)

    center_seed_a, cost_a = solve_ik(
        model, data, JOINT_SLICE_A, site_a,
        center_target, enforce_orientation=False,
        q_seed=home_qpos_a, arm="A",
    )
    center_seed_b, cost_b = solve_ik(
        model, data, JOINT_SLICE_B, site_b,
        center_target, enforce_orientation=False,
        q_seed=home_qpos_b, arm="B",
    )
    print(f"[init] Center seed IK — A: {cost_a:.4f}  B: {cost_b:.4f}")

    goal_pos = zone_pos(scene, task.goal_zone)

    print(f"[init] handoff_pos : {handoff_pos}")
    print(f"[init] goal_pos    : {goal_pos}")

    obstacle_body_id = -1
    if task.obstacle_name:
        oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, task.obstacle_name)
        if oid >= 0:
            obstacle_body_id = oid

    phase_queue      = list(task.phases)
    reward_log       = []
    phase_idx        = 0
    steps_in_phase   = 0
    q_phase_target   = None
    ee_target        = None
    current_pf       = None
    decompose_counts = {}

    with mujoco.viewer.launch_passive(
        model, data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:

        while viewer.is_running():

            if phase_idx >= len(phase_queue):
                data.ctrl[GRIP_IDX_A] = GRIP_OPEN
                data.ctrl[GRIP_IDX_B] = GRIP_OPEN
                mujoco.mj_step(model, data)
                viewer.sync()
                continue

            phase = phase_queue[phase_idx]
            arm   = phase.arm

            joint_slice = JOINT_SLICE_A if arm == "A" else JOINT_SLICE_B
            site_id     = site_a if arm == "A" else site_b
            grip_idx    = GRIP_IDX_A if arm == "A" else GRIP_IDX_B

            if current_pf is None:
                scene_state = build_scene_state(data, scene, handoff_pos, goal_pos)
                current_pf  = build_phase_functions(phase, scene_state)

                if phase.action == "retreat":
                    home_q = home_qpos_a if arm == "A" else home_qpos_b
                    q_phase_target = home_q.copy()
                    ik_cost        = 0.0
                    ee_target      = left_gripper_init_pos.copy() if arm  == "A" else right_gripper_init_pos.copy()

                    _home_q = home_q.copy()
                    _js     = joint_slice

                    def _retreat_reward(data, sa, sb, ob, obst, hp, gp):
                        return 1.0

                    def _retreat_is_done(data, sa, sb, ob, obst, hp, gp):
                        q_now = data.qpos[_js]
                        return bool(np.linalg.norm(q_now - _home_q) < 0.15)

                    def _retreat_gripper(data, sa, sb, ob, obst, hp, gp):
                        return GRIP_OPEN

                    current_pf = PhaseFunctions(
                        phase, _retreat_reward, _retreat_is_done, _retreat_gripper,
                        source="retreat"
                    )
                    print(
                        f"[runner] Phase {phase.phase_id} ({arm} retreat): "
                        f"driving to home config"
                    )

                if phase.action == "grasp":
                    # grasp only moves fingers — hold arm at current ctrl position
                    q_phase_target = (
                        data.ctrl[0:7].copy() if arm == "A" else data.ctrl[7:14].copy()
                    )
                    ik_cost   = 0.0
                    ee_target = data.site_xpos[site_id].copy()
                else:
                    ee_target = get_ee_target(phase, scene, data, handoff_pos, goal_pos)

                    if phase.action == "place":
                        q_seed = home_qpos_a if arm == "A" else home_qpos_b
                    elif phase.target in ("center", "handoff_zone"):
                        q_seed = center_seed_a if arm == "A" else center_seed_b
                    elif phase.target == "box":
                        # box could be anywhere, use center seed if box is near center,
                        # home seed if box is near own table
                        box_pos_now = get_box_pos(data, scene)
                        if abs(box_pos_now[0]) < 0.5:
                            q_seed = center_seed_a if arm == "A" else center_seed_b
                        else:
                            q_seed = home_qpos_a if arm == "A" else home_qpos_b
                    else:
                        q_seed = home_qpos_a if arm == "A" else home_qpos_b

                    q_phase_target, ik_cost = solve_ik_with_waypoints(
                        model, data, joint_slice, site_id, ee_target,
                        enforce_orientation=(phase.action == "reach"),
                        q_seed=q_seed,
                        arm=arm,
                    )
                    if ik_cost > 0.1 and phase.action == "reach":
                        q_no_orient, cost_no_orient = solve_ik_with_waypoints(
                            model, data, joint_slice, site_id, ee_target,
                            enforce_orientation=False,
                            q_seed=q_seed,
                            arm=arm,
                        )
                        if cost_no_orient < ik_cost:
                            print(
                                f"[runner] Orientation IK failed ({ik_cost:.4f}), "
                                f"using position-only ({cost_no_orient:.4f})"
                            )
                            q_phase_target = q_no_orient
                            ik_cost        = cost_no_orient

                print(
                    f"[runner] Phase {phase.phase_id} ({arm} {phase.action}): "
                    f"source={current_pf.source} IK={ik_cost:.4f} "
                    f"target={np.round(ee_target, 3)} "
                    f"dist={np.linalg.norm(data.site_xpos[site_id] - ee_target):.3f}"
                )

            q_ctrl_now = data.ctrl[0:7].copy() if arm == "A" else data.ctrl[7:14].copy()
            delta      = q_phase_target - q_ctrl_now
            q_cmd      = q_ctrl_now + np.clip(delta, -MAX_CTRL_DELTA, MAX_CTRL_DELTA)

            if arm == "A":
                data.ctrl[0:7]  = q_cmd
            else:
                data.ctrl[7:14] = q_cmd

            grip_val = current_pf.gripper(
                data, site_a, site_b,
                scene.box_body, obstacle_body_id,
                handoff_pos, goal_pos,
            )
            data.ctrl[grip_idx] = grip_val

            reward = current_pf.reward(
                data, site_a, site_b,
                scene.box_body, obstacle_body_id,
                handoff_pos, goal_pos,
            )
            reward_log.append((phase.phase_id, reward))

            steps_in_phase += 1

            done = current_pf.is_done(
                data, site_a, site_b,
                scene.box_body, obstacle_body_id,
                handoff_pos, goal_pos,
            )

            if done:
                print(
                    f"[runner] Phase {phase.phase_id} ({arm} {phase.action}) done: "
                    f"reward={reward:.3f} steps={steps_in_phase}"
                )

                if phase.action == "grasp" and not weld_active:
                    weld_id = weld_left_id if arm == "A" else weld_right_id
                    if weld_id >= 0:
                        enable_weld(model, data, weld_id)
                        weld_active = True
                        print(f"[runner] Weld activated for arm {arm}")

                if phase.action == "place" and weld_active:
                    disable_weld(model, data, weld_left_id)
                    disable_weld(model, data, weld_right_id)
                    weld_active = False
                    print(f"[runner] Weld released — place done for arm {arm}")

                # after arm A places at center, inject a retreat so it clears
                # the handoff zone before arm B tries to reach the box
                if (phase.action == "place"
                        and arm == "A"
                        and phase.target in ("center", "handoff_zone")):
                    retreat = Phase(
                        phase_id=phase.phase_id * 10 + 9,
                        arm="A",
                        action="retreat",
                        target="A",
                        depends_on=None,
                    )
                    phase_queue.insert(phase_idx + 1, retreat)
                    print(f"[runner] Inserted arm A retreat phase after place")

                phase_idx      += 1
                steps_in_phase  = 0
                current_pf      = None
                q_phase_target  = None
                ee_target       = None

            elif steps_in_phase >= MAX_STEPS_PER_PHASE:
                actual_ee = data.site_xpos[site_id]
                print(
                    f"[runner] Phase {phase.phase_id} ({arm} {phase.action}) timed out "
                    f"after {steps_in_phase} steps. reward={reward:.3f} "
                    f"EE={np.round(actual_ee, 3)} target={np.round(ee_target, 3)} "
                    f"dist={np.linalg.norm(actual_ee - ee_target):.3f}"
                )

                root_id = phase.phase_id
                count   = decompose_counts.get(root_id, 0)

                if phase.phase_id < 100 and count < MAX_DECOMPOSE_ATTEMPTS:
                    sub_phases = decompose_phase(phase, reason="timeout")
                    if sub_phases:
                        phase_queue = (
                            phase_queue[:phase_idx]
                            + sub_phases
                            + phase_queue[phase_idx+1:]
                        )
                        decompose_counts[root_id] = count + 1
                        print(
                            f"[runner] Decomposed phase {root_id} into "
                            f"{len(sub_phases)} sub-phases at position {phase_idx}"
                        )
                    else:
                        print(f"[runner] Decompose failed, skipping phase {phase.phase_id}")
                        phase_idx += 1
                else:
                    print(f"[runner] Skipping phase {phase.phase_id}")
                    phase_idx += 1

                if weld_active:
                    disable_weld(model, data, weld_left_id)
                    disable_weld(model, data, weld_right_id)
                    weld_active = False
                    print(f"[runner] Weld released on phase skip")

                steps_in_phase = 0
                current_pf     = None
                q_phase_target = None
                ee_target      = None

            mujoco.mj_step(model, data)

            if phase.action == "place" and weld_active:
                box_pos_now = get_box_pos(data, scene)
                target_pos  = handoff_pos if phase.target in ("center", "handoff_zone") else goal_pos
                if np.linalg.norm(box_pos_now[:2] - target_pos[:2]) < 0.08:
                    disable_weld(model, data, weld_left_id)
                    disable_weld(model, data, weld_right_id)
                    weld_active = False
                    print(f"[runner] Weld released — box arrived at target")

            viewer.sync()

    OUTPUTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    steps   = list(range(len(reward_log)))
    rewards = [r for _, r in reward_log]
    phases  = [p for p, _ in reward_log]

    ax.plot(steps, rewards, linewidth=0.8, color="#4a9eff")

    colors = ["#ff6b6b", "#4ecdc4", "#ffe66d",
            "#a8e6cf", "#ff8b94", "#c3a6ff"]

    phase_order = []
    seen = set()

    for pid in phases:
        if pid not in seen:
            seen.add(pid)
            phase_order.append(pid)

    for pid in phase_order:
        idxs = [i for i, p in enumerate(phases) if p == pid]

        if not idxs:
            continue

        # 1,100,101 -> root phase 1
        # 2,200,201 -> root phase 2
        root_pid = pid if pid < 100 else pid // 100

        color = colors[(root_pid - 1) % len(colors)]

        ax.axvspan(
            idxs[0],
            idxs[-1],
            alpha=0.15,
            color=color,
            label=f"Phase {pid}",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Per-phase Reward Curve")
    ax.legend(fontsize=7)
    plt.tight_layout()

    out = OUTPUTS_DIR / "reward_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nReward curve saved to {out}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="move box from table A to table B")
    args = parser.parse_args()

    run(args.task)
