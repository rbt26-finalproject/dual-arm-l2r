"""
Reach demo with box teleporting between table A and table B at random positions.

Both arms sweep their full horizontal reach while the red box teleports
every few seconds between the left (A) and right (B) side tables,
landing at a random position on whichever table it moves to.

Setup (one-time, run from repo root):
    mkdir aloha_menagerie && cd aloha_menagerie
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git .
    git sparse-checkout set aloha
    cd ..

Run:
    python reach_demo.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import importlib
from pathlib import Path

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aloha import base as aloha_base
from mujoco_playground._src.manipulation.aloha import aloha_constants as consts
from etils import epath

MENAGERIE_PATH = Path(__file__).parent / "aloha_menagerie"

if not MENAGERIE_PATH.exists():
    raise FileNotFoundError(
        f"Menagerie not found at {MENAGERIE_PATH}. "
        "Run the setup commands in the docstring first."
    )

mjx_env.MENAGERIE_PATH = epath.Path(str(MENAGERIE_PATH))
importlib.reload(aloha_base)

SCENE_XML = Path(__file__).parent / "assets" / "scene_dual_arm.xml"

BOX_QPOS_ADR = 16  # freejoint starts at qpos index 16

# Table A (left) and table B (right) center x positions
TABLE_A_X = -0.96
TABLE_B_X =  0.96

# Box sits at z=0.035 on the side tables (surface at z=0.005, box half-height=0.03)
BOX_Z = 0.035

# Random y range within table depth (table half-size y=0.37, keep box away from edges)
BOX_Y_RANGE = (-0.25, 0.25)

# Random x offset within table half-size (0.35), keep away from edges
BOX_X_OFFSET_RANGE = (-0.20, 0.20)

# How often the box teleports (seconds)
TELEPORT_INTERVAL = 3.0

# Arm sweep settings
SHOULDER_EXTENDED = -1.8
ELBOW_EXTENDED    = -1.36
GRIPPER_OPEN      =  0.037
SWEEP_FREQ        =  0.08

ACT_L = slice(0, 7)
ACT_R = slice(7, 14)


def load_model():
    assets  = aloha_base.get_assets()
    xml_str = SCENE_XML.read_text()
    return mujoco.MjModel.from_xml_string(xml_str, assets)


def extended_ctrl(waist):
    return np.array([waist, SHOULDER_EXTENDED, ELBOW_EXTENDED, 0.0, 0.0, 0.0, GRIPPER_OPEN])


def teleport_box(data, rng, current_table):
    next_table = "B" if current_table == "A" else "A"
    center_x   = TABLE_A_X if next_table == "A" else TABLE_B_X
    x = center_x + rng.uniform(*BOX_X_OFFSET_RANGE)
    y = rng.uniform(*BOX_Y_RANGE)
    data.qpos[BOX_QPOS_ADR    ] = x
    data.qpos[BOX_QPOS_ADR + 1] = y
    data.qpos[BOX_QPOS_ADR + 2] = BOX_Z
    data.qpos[BOX_QPOS_ADR + 3] = 1  # quaternion w
    data.qpos[BOX_QPOS_ADR + 4] = 0
    data.qpos[BOX_QPOS_ADR + 5] = 0
    data.qpos[BOX_QPOS_ADR + 6] = 0
    # zero out box velocity so it doesn't carry momentum from previous position
    data.qvel[BOX_QPOS_ADR - 1: BOX_QPOS_ADR + 5] = 0
    print(f"Box -> table {next_table}  ({x:.3f}, {y:.3f}, {BOX_Z})")
    return next_table


def main():
    model = load_model()
    data  = mujoco.MjData(model)
    rng   = np.random.default_rng(seed=42)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Start box on table A
    current_table = "A"
    data.qpos[BOX_QPOS_ADR    ] = TABLE_A_X + rng.uniform(*BOX_X_OFFSET_RANGE)
    data.qpos[BOX_QPOS_ADR + 1] = rng.uniform(*BOX_Y_RANGE)
    data.qpos[BOX_QPOS_ADR + 2] = BOX_Z
    data.qpos[BOX_QPOS_ADR + 3] = 1
    data.qpos[BOX_QPOS_ADR + 4] = 0
    data.qpos[BOX_QPOS_ADR + 5] = 0
    data.qpos[BOX_QPOS_ADR + 6] = 0
    mujoco.mj_forward(model, data)

    print("Viewer open. Box teleports every 3 seconds between table A and table B.")
    print("Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data,
                                      show_left_ui=False,
                                      show_right_ui=False) as viewer:
        t             = 0.0
        last_teleport = 0.0

        while viewer.is_running():
            step_start = time.perf_counter()

            waist_l = np.pi * np.sin(2 * np.pi * SWEEP_FREQ * t)
            waist_r = np.pi * np.sin(2 * np.pi * SWEEP_FREQ * t + np.pi)

            data.ctrl[ACT_L] = extended_ctrl(waist_l)
            data.ctrl[ACT_R] = extended_ctrl(waist_r)

            mujoco.mj_step(model, data)
            t += model.opt.timestep

            if t - last_teleport >= TELEPORT_INTERVAL:
                current_table = teleport_box(data, rng, current_table)
                mujoco.mj_forward(model, data)
                last_teleport = t

            viewer.sync()

            elapsed = time.perf_counter() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
