"""
Trajectory demo for Arm A Phase 1: reach toward the box on table A.

Parses the task using motion_descriptor (Gemini or rule-based fallback),
builds the reward pipeline (Gemini-generated or hardcoded fallback),
solves IK for the box position, then interpolates Arm A from home to
the reach target. Reward is printed live. Arm B stays at home.
Close the viewer window to exit.

Setup (one-time, run from repo root):
    mkdir aloha_menagerie && cd aloha_menagerie
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git .
    git sparse-checkout set aloha
    cd ..

Run:
    export GEMINI_API_KEY=api_key   # optional, falls back to rule-based
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

from motion_descriptor import parse_task
from reward_function import RewardPipeline

MENAGERIE_PATH = Path(__file__).parent.parent / "aloha_menagerie"
SCENE_XML      = Path(__file__).parent.parent / "assets" / "scene_dual_arm.xml"

if not MENAGERIE_PATH.exists():
    raise FileNotFoundError(
        f"Menagerie not found at {MENAGERIE_PATH}. "
        "Run the setup commands in the docstring first."
    )

mjx_env.MENAGERIE_PATH = epath.Path(str(MENAGERIE_PATH))
importlib.reload(aloha_base)

TASK_STR     = "move box from table A to table B"
BOX_QPOS_ADR = 16
BOX_START    = np.array([-0.96, 0.0, 0.035])
HANDOFF_POS  = np.array([0.0,   0.0, 0.05])
GOAL_POS     = np.array([0.96,  0.0, 0.035])
REACH_OFFSET = np.array([0.0,   0.0, 0.045])  # slightly above box surface
TRAJ_STEPS   = 2000

ACT_L = slice(0, 7)
ACT_R = slice(7, 14)


def load_model():
    assets  = aloha_base.get_assets()
    xml_str = SCENE_XML.read_text()
    return mujoco.MjModel.from_xml_string(xml_str, assets)


def solve_ik(model, data, site_id, target_pos):
    lo = model.jnt_range[:6, 0]
    hi = model.jnt_range[:6, 1]
    q0 = data.qpos[:6].copy()

    def cost(q):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        return float(np.linalg.norm(data.site_xpos[site_id] - target_pos))

    res = minimize(cost, q0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                   options={"maxiter": 500, "ftol": 1e-8})
    data.qpos[:6] = q0
    mujoco.mj_forward(model, data)
    return res.x, res.fun


def main():
    print(f"Task: {TASK_STR}")
    task     = parse_task(TASK_STR)
    print(task.describe())
    print()

    pipeline = RewardPipeline(task)
    print(pipeline.describe())
    print()

    model = load_model()
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    data.qpos[BOX_QPOS_ADR:BOX_QPOS_ADR + 3] = BOX_START
    data.qpos[BOX_QPOS_ADR + 3]               = 1
    data.qpos[BOX_QPOS_ADR + 4:BOX_QPOS_ADR + 7] = 0
    mujoco.mj_forward(model, data)

    site_a_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
    site_b_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")
    box_body_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    obstacle_body_id = box_body_id  # no separate obstacle in this demo

    print("Solving IK for box reach position...")
    reach_target  = BOX_START + REACH_OFFSET
    q_reach, dist = solve_ik(model, data, site_a_id, reach_target)
    print(f"IK solved: residual dist={dist:.6f}  target={reach_target}\n")

    q_home     = data.qpos[:6].copy()
    trajectory = np.linspace(q_home, q_reach, TRAJ_STEPS)
    HOME_R     = data.ctrl[ACT_R].copy()

    # Phase 1 in this task is "reach" for the box (first phase of task decomposition)
    active_phase = task.phases[0].phase_id

    print(f"Phase {active_phase}: Arm A moving toward box on table A.")
    print("Reward and EE position printed every second. Close the viewer to exit.\n")

    traj_idx   = 0
    phase_done = False
    last_print = 0.0

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
                    print(f"\nPhase {active_phase} trajectory complete at t={t:.1f}s")

            data.ctrl[ACT_R] = HOME_R
            mujoco.mj_step(model, data)
            t += model.opt.timestep

            if t - last_print >= 1.0:
                ee_a = data.site_xpos[site_a_id]
                dist = float(np.linalg.norm(ee_a - reach_target))
                r    = pipeline.compute(
                    phase_id=active_phase,
                    data=data,
                    site_a_id=site_a_id,
                    site_b_id=site_b_id,
                    object_body_id=box_body_id,
                    obstacle_body_id=obstacle_body_id,
                    handoff_pos=HANDOFF_POS,
                    goal_pos=GOAL_POS,
                )
                status = "done" if phase_done else f"step {traj_idx}/{TRAJ_STEPS}"
                print(f"t={t:.1f}s [{status}]  dist={dist:.4f}  "
                      f"reward={r:.4f}  "
                      f"EE_A=({ee_a[0]:.3f},{ee_a[1]:.3f},{ee_a[2]:.3f})")
                last_print = t

            viewer.sync()

            elapsed = time.perf_counter() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
