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
from reward_function import build_phase_functions, decompose_phase, GRIP_OPEN, GRIP_CLOSED
from handoff_resolver import compute_handoff_zone


MENAGERIE_PATH = Path(__file__).parent.parent / "aloha_menagerie"
SCENE_XML      = Path(__file__).parent.parent / "assets" / "scene_dual_arm.xml"
OUTPUTS_DIR    = Path(__file__).parent.parent / "outputs"

JOINT_SLICE_A = slice(0, 6)
JOINT_SLICE_B = slice(8, 14)

GRIP_IDX_A = 6
GRIP_IDX_B = 13

MIN_EE_Z = 0.05

MAX_STEPS_PER_PHASE    = 1200
MAX_DECOMPOSE_ATTEMPTS = 1
MAX_CTRL_DELTA         = 0.005


# -------------------------
# MODEL LOADING
# -------------------------
def load_model():
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
    if zone == "A":
        return np.array([scene.table_a_x, 0.0, scene.box_rest_z])
    if zone == "B":
        return np.array([scene.table_b_x, 0.0, scene.box_rest_z])
    if zone in ("center", "handoff_zone"):
        return np.array([scene.handoff_x, 0.0, scene.handoff_z])
    raise ValueError(f"Unknown zone: {zone}")


def set_box_pose(model, data, scene, pos):
    adr = scene.box_qpos_adr
    data.qpos[adr:adr+3]   = pos
    data.qpos[adr+3]        = 1.0
    data.qpos[adr+4:adr+7] = 0.0
    data.qvel[:]            = 0
    mujoco.mj_forward(model, data)


def get_box_pos(data, scene):
    adr = scene.box_qpos_adr
    return data.qpos[adr:adr+3].copy()


def build_scene_state(data, scene, handoff_pos, goal_pos):
    return {
        "box_pos":     get_box_pos(data, scene),
        "handoff_pos": handoff_pos,
        "goal_pos":    goal_pos,
        "ee_a":        data.site_xpos[scene.left_gripper].copy(),
        "ee_b":        data.site_xpos[scene.right_gripper].copy(),
        "ee_a_xmat":   data.site_xmat[scene.left_gripper].reshape(3, 3).copy(),
        "ee_b_xmat":   data.site_xmat[scene.right_gripper].reshape(3, 3).copy(),
        "min_ee_z":    MIN_EE_Z,
    }


# -------------------------
# IK SOLVER
# -------------------------
def solve_ik(model, data, joint_slice, site_id, ee_target, enforce_orientation=False):
    lo       = model.jnt_range[joint_slice, 0]
    hi       = model.jnt_range[joint_slice, 1]
    q0       = data.qpos[joint_slice].copy()
    q_backup = data.qpos.copy()
    v_backup = data.qvel.copy()

    def cost(q):
        data.qpos[joint_slice] = q
        mujoco.mj_forward(model, data)

        pos_err = np.linalg.norm(data.site_xpos[site_id] - ee_target)

        if not enforce_orientation:
            return float(pos_err)

        R = data.site_xmat[site_id].reshape(3, 3)

        # approach axis must be horizontal — penalize z component
        approach_tilt_err = abs(float(R[:, 0][2]))

        # Z axis must point up — keeps fingers parallel to table
        up_err = 1.0 - float(np.dot(R[:, 2], np.array([0.0, 0.0, 1.0])))

        # approach axis must point toward target in XY plane
        to_target   = ee_target - data.site_xpos[site_id]
        to_target_h = to_target.copy()
        to_target_h[2] = 0.0
        norm = np.linalg.norm(to_target_h)
        if norm > 1e-6:
            to_target_h /= norm
            approach_h        = R[:, 0].copy()
            approach_h[2]     = 0.0
            approach_h_norm   = np.linalg.norm(approach_h)
            if approach_h_norm > 1e-6:
                approach_h /= approach_h_norm
            approach_dir_err = 1.0 - float(np.dot(approach_h, to_target_h))
        else:
            approach_dir_err = 0.0

        return float(pos_err
                     + 0.5 * up_err
                     + 0.5 * approach_tilt_err
                     + 0.3 * approach_dir_err)

    res = minimize(
        cost, q0,
        method="L-BFGS-B",
        bounds=list(zip(lo, hi)),
        options={"maxiter": 800, "ftol": 1e-10},
    )

    data.qpos[joint_slice] = res.x
    mujoco.mj_forward(model, data)
    actual_cost = float(np.linalg.norm(data.site_xpos[site_id] - ee_target))

    data.qpos[:] = q_backup
    data.qvel[:] = v_backup
    mujoco.mj_forward(model, data)

    return np.clip(res.x, lo, hi), actual_cost


# -------------------------
# EE TARGET FOR IK
# -------------------------
def get_ee_target(phase, scene, data, handoff_pos, goal_pos):
    """
    Returns the IK position target for the gripper site.
    Must stay consistent with what _resolve_target in reward_function computes
    so is_done fires when the arm actually reaches the target.
    """
    arm    = phase.arm
    action = phase.action
    target = phase.target

    def clamp_z(t):
        t = t.copy()
        t[2] = max(t[2], MIN_EE_Z)
        return t

    if action == "reach":
        if target == "box":
            t = get_box_pos(data, scene).copy()
            t[2] = max(t[2], MIN_EE_Z)
            return clamp_z(t)
        if target in ("center", "handoff_zone"):
            t = handoff_pos.copy()
            t[2] = max(t[2] + 0.05, MIN_EE_Z)
            return clamp_z(t)
        if target in ("A", "B"):
            t = zone_pos(scene, target).copy()
            t[2] = max(t[2] + 0.05, MIN_EE_Z)
            return clamp_z(t)
        t = handoff_pos.copy()
        t[2] = max(t[2] + 0.05, MIN_EE_Z)
        return clamp_z(t)

    if action == "place":
        if arm == "A":
            t = handoff_pos.copy()
            t[2] = max(t[2] + 0.05, MIN_EE_Z)
            return clamp_z(t)
        t = goal_pos.copy()
        t[2] = max(t[2] + 0.05, MIN_EE_Z)
        return clamp_z(t)

    if action == "clear_obstacle":
        t = data.xpos[scene.box_body].copy()
        t[2] = max(t[2] + 0.10, MIN_EE_Z)
        return clamp_z(t)

    t = handoff_pos.copy()
    t[2] = max(t[2] + 0.05, MIN_EE_Z)
    return clamp_z(t)


# -------------------------
# MAIN RUN
# -------------------------
def run(task_str):
    print(f"\nTask: {task_str}")

    task = parse_task(task_str)
    print("\n=== TASK PHASES ===")
    for p in task.phases:
        print(f"  phase={p.phase_id} arm={p.arm} action={p.action} target={p.target}")
    print("===================\n")

    model = load_model()
    data  = mujoco.MjData(model)
    scene = SceneInfo(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    start_pos = np.array([
        scene.table_a_x if task.start_zone == "A" else scene.table_b_x,
        0.0,
        scene.box_rest_z,
    ])
    set_box_pose(model, data, scene, start_pos)

    print(f"[init] box pos     : {get_box_pos(data, scene)}")
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
    goal_pos    = zone_pos(scene, task.goal_zone)

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

                # grasp only moves fingers — hold arm at current ctrl position
                if phase.action == "grasp":
                    q_phase_target = (
                        data.ctrl[0:6].copy() if arm == "A" else data.ctrl[8:14].copy()
                    )
                    ik_cost   = 0.0
                    ee_target = data.site_xpos[site_id].copy()
                else:
                    ee_target      = get_ee_target(phase, scene, data, handoff_pos, goal_pos)
                    q_phase_target, ik_cost = solve_ik(
                        model, data, joint_slice, site_id, ee_target,
                        enforce_orientation=(phase.action == "reach"),
                    )

                print(
                    f"[runner] Phase {phase.phase_id} ({arm} {phase.action}): "
                    f"source={current_pf.source} IK={ik_cost:.4f} "
                    f"target={np.round(ee_target, 3)} "
                    f"dist={np.linalg.norm(data.site_xpos[site_id] - ee_target):.3f}"
                )

            q_ctrl_now = data.ctrl[0:6].copy() if arm == "A" else data.ctrl[8:14].copy()
            delta      = q_phase_target - q_ctrl_now
            q_cmd      = q_ctrl_now + np.clip(delta, -MAX_CTRL_DELTA, MAX_CTRL_DELTA)

            if arm == "A":
                data.ctrl[0:6]  = q_cmd
            else:
                data.ctrl[8:14] = q_cmd

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

                steps_in_phase = 0
                current_pf     = None
                q_phase_target = None
                ee_target      = None

            mujoco.mj_step(model, data)
            viewer.sync()

    # -------------------------
    # REWARD CURVE PLOT
    # -------------------------
    OUTPUTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    steps   = list(range(len(reward_log)))
    rewards = [r for _, r in reward_log]
    phases  = [p for p, _ in reward_log]

    ax.plot(steps, rewards, linewidth=0.8, color="#4a9eff")

    colors = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a8e6cf", "#ff8b94", "#c3a6ff"]
    for pid in sorted(set(phases)):
        idxs = [i for i, p in enumerate(phases) if p == pid]
        if idxs:
            ax.axvspan(
                idxs[0], idxs[-1],
                alpha=0.15,
                color=colors[(pid - 1) % len(colors)],
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
