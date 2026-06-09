# Language-to-Reward Driven Dual-Arm Manipulation in MuJoCo

A robotics final project for Fasilkom UI Robotics 2026. Uses a single LLM agent (Gemini) to control a dual-arm ALOHA robot in MuJoCo, translating natural language instructions into executable manipulation phases via a two-stage language-to-reward pipeline.

---

## Overview

Given a task like `"move the box from table A to table B"`, the system:
1. Parses the instruction into a structured motion plan
2. Decomposes it into ordered arm phases (reach → grasp → place)
3. Generates per-phase reward, completion, and gripper functions
4. Executes the phases in MuJoCo using IK-based joint control

The approach is inspired by [Language to Rewards (Yu et al., 2023)](https://language-to-reward.github.io/) and extends it to a dual-arm handoff setting.

---

## Project Structure

```
.
├── src/
│   ├── mujoco_runner.py       # Main entry point, simulation loop, IK solver
│   ├── motion_descriptor.py   # Stage 1 & 2: task string → TaskDecomposition
│   ├── reward_function.py     # Per-phase reward/is_done/gripper generation
│   └── handoff_resolver.py    # Computes the center handoff zone position
├── assets/
│   └── scene_dual_arm.xml     # MuJoCo scene with dual-arm ALOHA + side tables
├── aloha_menagerie/           # Aloha robot model assets
└── outputs/
    └── reward_curve.png       # Saved per-phase reward plot after each run
```

---

## Scene

Based on the `mjx_hand_over` scene from MuJoCo Menageries (Aloha environment), modified with two side tables:

- **Table A** — left side, reachable only by Arm A
- **Table B** — right side, reachable only by Arm B
- **Center table** — reachable by both arms, used as handoff zone

---

## Pipeline

### Stage 1 — Motion Descriptor (`motion_descriptor.py`)

Gemini converts a free-form task string into a constrained structured plan:

```
[start of plan]
Object to manipulate: box
Start location: table_A
Goal location: table_B
Handoff required: yes
Arm A reaches the object.
Arm A grasps the object.
Arm A places the object at center.
Arm B reaches the object.
Arm B grasps the object.
Arm B places the object at goal location.
[end of plan]
```

### Stage 2 — Task Decomposer (`motion_descriptor.py`)

The structured plan is converted to a JSON `TaskDecomposition` with ordered `Phase` objects: `{phase_id, arm, action, target, depends_on}`.

### Stage 3 — Reward Coder (`reward_function.py`)

For each phase, Gemini produces calls to a small constrained API:

```python
set_ee_target("box")
set_gripper("open")
set_done_threshold(0.075)
```

These are executed against a `PhaseConfig` object that builds the `reward()`, `is_done()`, and `gripper_command()` closures used by the runner. All geometry and numpy math stays in Python — Gemini never outputs raw math.

### Execution — Runner (`mujoco_runner.py`)

Each phase drives one arm via SLSQP-based IK with:
- Table collision avoidance (minimum EE z constraint)
- Robot base exclusion zone
- Optional orientation enforcement on reach phases
- Rate-limited joint control (`MAX_CTRL_DELTA = 0.008` per step)
- Weld constraint activation on grasp, release on place
- Automatic phase decomposition if a phase times out

---

## Fallbacks

Both the motion descriptor and reward coder fall back gracefully when `GEMINI_API_KEY` is not set or an API call fails:

- `motion_descriptor.py` — rule-based keyword parser extracts zones, object, and obstacle from the task string
- `reward_function.py` — hardcoded `PhaseConfig` per action type (`reach`/`grasp`/`place`)

---

## Requirements

- Python 3.10+
- `mujoco`
- `mujoco-playground`
- `scipy`
- `matplotlib`
- `google-genai`

---

## Usage

```bash
# Default task
python src/mujoco_runner.py

# Custom task
python src/mujoco_runner.py --task "move box from table B to table A"
```

Set your Gemini API key to enable LLM-based parsing and reward generation:

```bash
export GEMINI_API_KEY=your_key_here
```

If the key is not set, the system runs fully on rule-based and hardcoded fallbacks.

---

## Output

After execution, a per-phase reward curve is saved to `outputs/reward_curve.png`. Each phase is color-coded as a background span, with the reward signal plotted over simulation steps.

---

## Limitations

- Limited task vocabulary (single object, fixed zones)
- Simplified manipulation assumptions (no true grasp physics, weld-based holding)
- Phases occasionally time out before completion, especially across the handoff

## Future Work

- Multi-object tasks
- More complex obstacle interactions
- Support for more than two arms
