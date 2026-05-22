"""
Reward function generator for the dual-arm task.

Follows the Reward Coder concept from Language to Rewards (Yu et al., 2023):
Gemini generates executable Python reward code for each phase based on the
TaskDecomposition from motion_descriptor.py. Falls back to hardcoded reward
functions when GEMINI_API_KEY is not set or generation fails.

The generated code is executed in a sandboxed local namespace and must define
a function with signature:
    def reward(data, site_a_id, site_b_id, object_body_id,
               obstacle_body_id, handoff_pos, goal_pos) -> float
"""

import os
import re
import time
import numpy as np
import mujoco

import google.genai as genai

from motion_descriptor import parse_task

REWARD_CODER_SYSTEM_PROMPT = """
You are a reward function coder for a dual-arm robot manipulation system in MuJoCo.

You will be given a phase description and must output ONLY executable Python code
that defines a single function named `reward` with this exact signature:

    def reward(data, site_a_id, site_b_id, object_body_id,
               obstacle_body_id, handoff_pos, goal_pos):

Available variables in scope: numpy as np, mujoco.
Available data fields:
    data.site_xpos[site_id]    -> (3,) array, end-effector position
    data.xpos[body_id]         -> (3,) array, body center position
    handoff_pos                -> (3,) array, handoff zone center
    goal_pos                   -> (3,) array, goal zone center

The function must return a float reward in (0.0, 1.0] using exponential decay:
    np.exp(-scale * distance)

Output ONLY the Python function code, no explanation, no markdown fences.
"""

PHASE_PROMPT_TEMPLATE = """
Phase {phase_id}: Arm {arm} performs action '{action}' on target '{target}'.

Write the reward function for this phase.
- For 'clear_obstacle': reward = how close EE_{arm} is to the obstacle center.
- For 'reach': reward = how close EE_{arm} is to the object center.
- For 'grasp': reward = how close EE_{arm} is to the object center (same as reach).
- For 'place' with target 'handoff_zone': reward = how close object is to handoff_pos (x-y only).
- For 'place' with target goal zone: reward = how close object is to goal_pos (x-y only).

Use site_a_id for Arm A, site_b_id for Arm B.
Use exp(-scale * dist) with scale=10.0 for EE-to-target and scale=8.0 for object-to-zone.
"""


def _call_gemini_reward(phase):
    """Asks Gemini to write reward code for a given phase. Returns code string or None."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        client = genai.Client(api_key=api_key)

        prompt = PHASE_PROMPT_TEMPLATE.format(
            phase_id=phase.phase_id,
            arm=phase.arm,
            action=phase.action,
            target=phase.target,
        )

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=REWARD_CODER_SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )

        print(response)

        text = getattr(response, "text", None)

        if not text:
            print(
                f"[reward_function] Gemini returned empty response "
                f"for phase {phase.phase_id}"
            )
            return None

        code = text.strip()

        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"\s*```$", "", code)

        return code

    except Exception as e:
        print(
            f"[reward_function] Gemini reward coder failed "
            f"for phase {phase.phase_id}: {e}"
        )

def _compile_reward_fn(code, phase_id):
    """Compiles generated reward code and returns the callable, or None on error."""
    try:
        namespace = {"np": np, "mujoco": mujoco}
        exec(code, namespace)
        fn = namespace.get("reward")
        if fn is None:
            print(f"[reward_function] Generated code for phase {phase_id} has no 'reward' function.")
            return None
        return fn
    except Exception as e:
        print(f"[reward_function] Failed to compile generated reward for phase {phase_id}: {e}")
        return None


# Hardcoded fallback reward functions

def _fallback_reward_phase(phase):
    """Returns a hardcoded reward function for a given phase as fallback."""
    arm    = phase.arm
    action = phase.action
    target = phase.target

    if action == "clear_obstacle":
        def reward(data, site_a_id, site_b_id, object_body_id,
                   obstacle_body_id, handoff_pos, goal_pos):
            ee  = data.site_xpos[site_a_id if arm == "A" else site_b_id]
            obs = data.xpos[obstacle_body_id]
            return float(np.exp(-10.0 * np.linalg.norm(ee - obs)))

    elif action in ("reach", "grasp"):
        def reward(data, site_a_id, site_b_id, object_body_id,
                   obstacle_body_id, handoff_pos, goal_pos):
            ee  = data.site_xpos[site_a_id if arm == "A" else site_b_id]
            obj = data.xpos[object_body_id]
            return float(np.exp(-10.0 * np.linalg.norm(ee - obj)))

    elif action == "place" and target == "handoff_zone":
        def reward(data, site_a_id, site_b_id, object_body_id,
                   obstacle_body_id, handoff_pos, goal_pos):
            obj = data.xpos[object_body_id]
            hp  = np.array(handoff_pos)
            return float(np.exp(-8.0 * np.linalg.norm(obj[:2] - hp[:2])))

    else:  # place at goal
        def reward(data, site_a_id, site_b_id, object_body_id,
                   obstacle_body_id, handoff_pos, goal_pos):
            obj = data.xpos[object_body_id]
            gp  = np.array(goal_pos)
            return float(np.exp(-8.0 * np.linalg.norm(obj[:2] - gp[:2])))

    return reward


class RewardPipeline:
    """
    Builds and holds the full set of reward functions for a TaskDecomposition.
    Tries Gemini for each phase, falls back to hardcoded if unavailable.
    """

    def __init__(self, task_decomposition):
        self.task  = task_decomposition
        self.fns   = {}  # phase_id -> callable
        self.sources = {}  # phase_id -> "gemini" or "fallback"
        self._build(task_decomposition.phases)

    def _build(self, phases):
        for phase in phases:
            code = _call_gemini_reward(phase)
            if code:
                fn = _compile_reward_fn(code, phase.phase_id)
                if fn:
                    self.fns[phase.phase_id]     = fn
                    self.sources[phase.phase_id] = "gemini"
                    print(f"[reward_function] Phase {phase.phase_id}: reward generated by Gemini.")
                    continue
            self.fns[phase.phase_id]     = _fallback_reward_phase(phase)
            self.sources[phase.phase_id] = "fallback"
            print(f"[reward_function] Phase {phase.phase_id}: using hardcoded fallback reward.")

    def compute(self, phase_id, data, site_a_id, site_b_id,
                object_body_id, obstacle_body_id, handoff_pos, goal_pos):
        fn = self.fns.get(phase_id)
        if fn is None:
            raise ValueError(f"No reward function for phase {phase_id}.")
        return fn(data, site_a_id, site_b_id, object_body_id,
                  obstacle_body_id, handoff_pos, goal_pos)

    def describe(self):
        lines = ["Reward pipeline:"]
        for pid, src in sorted(self.sources.items()):
            lines.append(f"  Phase {pid}: {src}")
        return "\n".join(lines)


if __name__ == "__main__":
    task = parse_task("move box behind obstacle from table A to table B")

    for phase in task.phases:
        _call_gemini_reward(phase)
        time.sleep(2)  # jeda 1 detik antar request

    pipeline = RewardPipeline(task)
    print(pipeline.describe())
