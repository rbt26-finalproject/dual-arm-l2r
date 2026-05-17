"""
Trajectory demo for Arm A Phase 1: end effector A reach toward the box on table A.

Solves IK for the box position, then interpolates Arm A from start position to
IK target in joint space. Reward is printed. Arm B doesnt do anything.
Close the viewer window to exit.

Setup (first time only and run from project root):
    mkdir aloha_menagerie && cd aloha_menagerie
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git .
    git sparse-checkout set aloha
    cd ..

Run:
    python src/trajectory_arm_a.py
"""

import sys
import time
import importlib
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aloha import base as aloha_base
from etils import epath

from reward_function import compute_reward

MENAGERIE_PATH = Path(__file__).parent.parent / "aloha_menagerie"
SCENE_XML      = Path(__file__).parent.parent / "assets" / "scene_dual_arm.xml"

if not MENAGERIE_PATH.exists():
    raise FileNotFoundError(
        f"Menagerie not found at {MENAGERIE_PATH}. "
        "Run the setup commands in the docstring first."
    )

mjx_env.MENAGERIE_PATH = epath.Path(str(MENAGERIE_PATH))
importlib.reload(aloha_base)

BOX_QPOS_ADR = 16
BOX_START    = np.array([-0.96, 0.0, 0.035])
HANDOFF_POS  = np.array([0.0,   0.0, 0.05])
GOAL_POS     = np.array([0.96,  0.0, 0.035])

ACT_L = slice(0, 7)
ACT_R = slice(7, 14)

HOME_R = np.array([0.083383, -0.554374, 1.23718, 0.108587, -0.65136, -0.072894, 0.037])

# Slightly above box surface so EE doesn't collide with table
REACH_TARGET_OFFSET = np.array([0.0, 0.0, 0.045])

# Number of sim steps to interpolate from home to reach target
TRAJ_STEPS = 2000


def load_model():
    assets  = aloha_base.get_assets()
    xml_str = SCENE_XML.read_text()
    return mujoco.MjModel.from_xml_string(xml_str, assets)


def solve_ik(model, data, site_id, target_pos):
    """Finds joint angles for the left arm that put the EE at target_pos."""
    lo = model.jnt_range[:6, 0]
    hi = model.jnt_range[:6, 1]
    q0 = data.qpos[:6].copy()

    def cost(q):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        return float(np.linalg.norm(data.site_xpos[site_id] - target_pos))

    res = minimize(cost, q0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                   options={"maxiter": 500, "ftol": 1e-8})

    data.qpos[:6] = q0  # restore
    mujoco.mj_forward(model, data)
    return res.x, res.fun


def main():
    model = load_model()
    data  = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    data.qpos[BOX_QPOS_ADR:BOX_QPOS_ADR + 3] = BOX_START
    data.qpos[BOX_QPOS_ADR + 3]               = 1
    data.qpos[BOX_QPOS_ADR + 4:BOX_QPOS_ADR + 7] = 0
    mujoco.mj_forward(model, data)

    site_a_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
    site_b_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")

    print("Solving IK for box position...")
    reach_target = BOX_START + REACH_TARGET_OFFSET
    q_reach, ik_dist = solve_ik(model, data, site_a_id, reach_target)
    print(f"IK solved: dist={ik_dist:.6f}  target={reach_target}")

    # Build joint-space trajectory: home -> reach via linear interpolation
    q_home = data.qpos[:6].copy()
    trajectory = np.linspace(q_home, q_reach, TRAJ_STEPS)

    print("Phase 1: Arm A moving toward box on table A.")
    print("Reward printed every second. Close the window to exit.")

    traj_idx  = 0
    last_print = 0.0
    phase_done = False

    with mujoco.viewer.launch_passive(model, data,
                                      show_left_ui=False,
                                      show_right_ui=False) as viewer:
        t = 0.0

        while viewer.is_running():
            step_start = time.perf_counter()

            if not phase_done:
                if traj_idx < len(trajectory):
                    data.ctrl[0:6] = trajectory[traj_idx]
                    traj_idx += 1
                else:
                    data.ctrl[0:6] = q_reach
                    phase_done = True

            data.ctrl[ACT_R] = HOME_R

            mujoco.mj_step(model, data)
            t += model.opt.timestep

            ee_a = data.site_xpos[site_a_id]
            dist = float(np.linalg.norm(ee_a - reach_target))
            reward = compute_reward(
                phase_id=2,  # reach object reward
                data=data,
                site_a_id=site_a_id,
                site_b_id=site_b_id,
                object_body_id=box_body_id,
                obstacle_body_id=box_body_id,
                handoff_pos=HANDOFF_POS,
                goal_pos=GOAL_POS,
            )

            if t - last_print >= 1.0:
                status = "done" if phase_done else f"step {traj_idx}/{TRAJ_STEPS}"
                print(f"t={t:.1f}s  [{status}]  "
                      f"dist_to_box={dist:.4f}  reward={reward:.4f}  "
                      f"EE_A=({ee_a[0]:.3f},{ee_a[1]:.3f},{ee_a[2]:.3f})")
                last_print = t

            if phase_done and t - last_print <= 0.01:
                print(f"\nPhase 1 complete. Arm A reached the box.  final reward={reward:.4f}")

            viewer.sync()

            elapsed = time.perf_counter() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
