"""
Reach demo for the Aloha dual-arm environment from MuJoCo Playground.

Both arms sweep their full waist range while held in an extended pose,
tracing out the maximum horizontal reach envelope of each arm.
Close the viewer window to exit.

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


def load_model():
    assets  = aloha_base.get_assets()
    xml_str = epath.Path((consts.XML_PATH / "mjx_hand_over.xml").as_posix()).read_text()
    return mujoco.MjModel.from_xml_string(xml_str, assets)


# Arm fully extended horizontally: shoulder pitched down, elbow straightened.
# This gives ~1.27m reach radius — the maximum horizontal envelope.
SHOULDER_EXTENDED = -1.8
ELBOW_EXTENDED    = -1.36
GRIPPER_OPEN      =  0.037

# Waist sweeps its full joint range ±3.14 rad so the extended arm traces a full arc.
WAIST_MIN = -3.14
WAIST_MAX =  3.14
SWEEP_FREQ = 0.08  # Hz, one full sweep every ~12 seconds

ACT_L = slice(0, 7)
ACT_R = slice(7, 14)


def extended_ctrl(waist):
    """Returns ctrl for one arm at given waist angle, arm fully extended."""
    return np.array([waist, SHOULDER_EXTENDED, ELBOW_EXTENDED, 0.0, 0.0, 0.0, GRIPPER_OPEN])


def main():
    model = load_model()
    data  = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    site_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
    site_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")

    print("Viewer open. Close the window to exit.")
    print("Both arms sweeping full horizontal reach (~1.27m radius).")

    with mujoco.viewer.launch_passive(model, data,
                                      show_left_ui=False,
                                      show_right_ui=False) as viewer:
        t    = 0.0
        step = 0
        while viewer.is_running():
            step_start = time.perf_counter()

            # Waist oscillates over full range; arms stay extended
            waist_mid   = (WAIST_MAX + WAIST_MIN) / 2
            waist_amp   = (WAIST_MAX - WAIST_MIN) / 2
            waist_l     = waist_mid + waist_amp * np.sin(2 * np.pi * SWEEP_FREQ * t)
            waist_r     = waist_mid + waist_amp * np.sin(2 * np.pi * SWEEP_FREQ * t + np.pi)  # opposite phase

            data.ctrl[ACT_L] = extended_ctrl(waist_l)
            data.ctrl[ACT_R] = extended_ctrl(waist_r)

            mujoco.mj_step(model, data)
            t    += model.opt.timestep
            step += 1

            if step % 400 == 0:
                ee_l = data.site_xpos[site_l]
                ee_r = data.site_xpos[site_r]
                r_l  = np.sqrt(ee_l[0]**2 + ee_l[1]**2)
                r_r  = np.sqrt(ee_r[0]**2 + ee_r[1]**2)
                print(f"t={t:.1f}s  "
                      f"Left  EE: ({ee_l[0]:.3f}, {ee_l[1]:.3f}, {ee_l[2]:.3f})  r={r_l:.3f}m  |  "
                      f"Right EE: ({ee_r[0]:.3f}, {ee_r[1]:.3f}, {ee_r[2]:.3f})  r={r_r:.3f}m")

            viewer.sync()

            elapsed = time.perf_counter() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
