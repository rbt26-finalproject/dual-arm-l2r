"""
reward_function.py

Generates per-phase reward(), is_done(), and gripper_command() functions
for a dual-arm robot manipulation task in MuJoCo.

Gemini is asked to produce calls to a small constrained API
(set_ee_target, set_gripper, set_done_threshold, set_alignment_weight,
set_alignment_threshold) rather than raw Python. The API functions are
implemented here using live scene state so all geometry stays in Python,
not in the LLM output.

Falls back to hardcoded phase logic when Gemini is unavailable or the
generated API calls fail to parse. Also handles phase decomposition,
splitting a timed-out phase into sub-phases via a separate Gemini call.
"""

import os
import re
import json
import time
import numpy as np
import mujoco
import google.genai as genai
from types_ import Phase


GRIP_OPEN   = 0.037
GRIP_CLOSED = 0.0


# -------------------------
# PROMPTS
# -------------------------

REWARD_SYSTEM_PROMPT = """
You are a reward function coder for a dual-arm robot manipulation system.

Output ONLY calls to the following API functions — no imports, no numpy, no math,
no other code.

Available API:

set_ee_target(target, z_offset=0.0)
  Sets the EE position target for reward and is_done.
  target must be one of: "box", "handoff_zone", "goal_zone"
  z_offset: additional height above the target center in meters (default 0.0).
  For reach/grasp to box: use z_offset=0.0.
  For handoff/goal: use z_offset=0.05 to clear the table surface.
  For Arm A place phases: always use "handoff_zone", never "goal_zone".
  For Arm B place phases: use "goal_zone".

set_done_threshold(distance)
  Sets the distance threshold in meters for is_done(). Default is 0.10.

set_alignment_weight(weight)
  Sets how much approach axis alignment is weighted in reward and is_done.
  0.0 = position only, 1.0 = full alignment penalty. Default is 0.5.

set_alignment_threshold(min_alignment)
  Sets minimum alignment dot product required for is_done(). Range -1.0 to 1.0.
  Default is 0.5. Set to -1.0 to disable alignment check.

set_gripper(state)
  state must be one of: "open", "closed", "open_when_done"
  "open"           -> always open  (reach, clear_obstacle)
  "closed"         -> always closed unconditionally (grasp)
  "open_when_done" -> closed while moving, opens when is_done() (place)

Rules:
1. Call set_ee_target() exactly once.
2. Call set_gripper() exactly once.
3. For grasp: set_gripper("closed") with no conditions.
4. For reach: set_gripper("open") with no conditions.
5. No other code. No explanation. No markdown.
"""

PHASE_PROMPT_TEMPLATE = """
=== SCENE STATE ===
box position:        {box_pos}
box center z:        {box_z:.4f}
box top z:           {box_top:.4f}
handoff center:      {handoff_pos}
goal position:       {goal_pos}
min reachable EE z:  {min_ee_z:.4f}
current EE position: {ee_current}

=== PHASE ===
Phase {phase_id}: Arm {arm} performs '{action}' targeting '{target}'.

=== TARGET GUIDE ===
- "box"          : EE moves to the box. Use z_offset=0.0 for both reach and grasp.
                   The IK orientation constraint ensures the gripper is horizontal
                   and fingers straddle the box sides.
- "handoff_zone" : EE moves to the handoff zone. Use z_offset=0.05.
- "goal_zone"    : EE moves to the goal zone. Use z_offset=0.05.
                   Only valid for Arm B place phases.
                   For Arm A place phases always use "handoff_zone".

Write ONLY API calls for this phase. No explanation. No markdown.
"""


# -------------------------
# PHASE CONFIG
# -------------------------

class PhaseConfig:
    """Holds the configuration produced by executing the Gemini reward API calls."""
    def __init__(self):
        self.target              = "box"
        self.z_offset            = 0.0
        self.done_threshold      = 0.10
        self.alignment_weight    = 0.8
        self.alignment_threshold = 0.8
        self.gripper_state       = "open"


# -------------------------
# FUNCTION BUILDER
# -------------------------

def _build_functions(config, phase, scene_state):
    """
    Builds reward(), is_done(), gripper_command() closures from a PhaseConfig.
    All geometry and numpy math lives here — nothing in the LLM output.
    """
    arm      = phase.arm
    min_ee_z = scene_state["min_ee_z"]

    target_key   = config.target
    z_offset     = config.z_offset
    done_thresh  = config.done_threshold
    align_weight = config.alignment_weight
    align_thresh = config.alignment_threshold
    grip_state   = config.gripper_state

    site_fn = (lambda sa, sb: sa) if arm == "A" else (lambda sa, sb: sb)

    def _resolve_target(data, object_body_id, hp, gp):
        if target_key == "box":
            t    = data.xpos[object_body_id].copy()
            t[2] = max(t[2] + z_offset, min_ee_z)
            return t
        if target_key == "handoff_zone":
            t    = hp.copy()
            t[2] = max(t[2] + max(z_offset, 0.05), min_ee_z)
            return t
        if target_key == "goal_zone":
            t    = gp.copy()
            t[2] = max(t[2] + max(z_offset, 0.05), min_ee_z)
            return t
        t    = hp.copy()
        t[2] = max(t[2] + 0.05, min_ee_z)
        return t

    def _alignment(data, site_id, target_pos):
        ee_pos    = data.site_xpos[site_id]
        to_target = target_pos - ee_pos
        norm      = np.linalg.norm(to_target)
        if norm < 1e-6:
            return 1.0
        approach = data.site_xmat[site_id].reshape(3, 3)[:, 0]
        return float(np.dot(approach, to_target / norm))

    def reward(data, site_a_id, site_b_id, object_body_id,
               obstacle_body_id, handoff_pos, goal_pos):
        sid   = site_fn(site_a_id, site_b_id)
        tgt   = _resolve_target(data, object_body_id, handoff_pos, goal_pos)
        dist  = np.linalg.norm(data.site_xpos[sid] - tgt)
        pos_r = float(np.exp(-10.0 * dist))
        if align_weight > 0.0:
            al = _alignment(data, sid, tgt)
            return pos_r * (1.0 - align_weight + align_weight * max(al, 0.0))
        return pos_r

    def is_done(data, site_a_id, site_b_id, object_body_id,
                obstacle_body_id, handoff_pos, goal_pos):
        sid  = site_fn(site_a_id, site_b_id)
        tgt  = _resolve_target(data, object_body_id, handoff_pos, goal_pos)
        dist = np.linalg.norm(data.site_xpos[sid] - tgt)
        if dist >= done_thresh:
            return False
        if align_weight > 0.0 and align_thresh > -1.0:
            return _alignment(data, sid, tgt) >= align_thresh
        return True

    def gripper_command(data, site_a_id, site_b_id, object_body_id,
                        obstacle_body_id, handoff_pos, goal_pos):
        if grip_state == "open":
            return GRIP_OPEN
        if grip_state == "closed":
            return GRIP_CLOSED
        sid  = site_fn(site_a_id, site_b_id)
        tgt  = _resolve_target(data, object_body_id, handoff_pos, goal_pos)
        dist = np.linalg.norm(data.site_xpos[sid] - tgt)
        return GRIP_OPEN if dist < done_thresh else GRIP_CLOSED

    return reward, is_done, gripper_command


# -------------------------
# GEMINI CALL
# -------------------------

def _call_gemini(phase, scene_state: dict):
    """
    Asks Gemini to produce reward API calls for this phase.
    Returns a populated PhaseConfig on success, None on failure.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        client = genai.Client(api_key=api_key)
        ee_key = "ee_a" if phase.arm == "A" else "ee_b"

        prompt = PHASE_PROMPT_TEMPLATE.format(
            box_pos=np.round(scene_state["box_pos"], 4).tolist(),
            box_z=float(scene_state["box_pos"][2]),
            box_top=float(scene_state["box_pos"][2]) + 0.035,
            handoff_pos=np.round(scene_state["handoff_pos"], 4).tolist(),
            goal_pos=np.round(scene_state["goal_pos"], 4).tolist(),
            min_ee_z=scene_state["min_ee_z"],
            ee_current=np.round(scene_state[ee_key], 4).tolist(),
            phase_id=phase.phase_id,
            arm=phase.arm,
            action=phase.action,
            target=phase.target,
        )

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=REWARD_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )

        print(f"\n[gemini reward api]\n{response.text}\n")
        return _parse_api_calls(response.text, phase)

    except Exception as e:
        print(f"[reward_function] Gemini call failed for phase {phase.phase_id}: {e}")
        return None


def _parse_api_calls(text, phase):
    """
    Executes Gemini-generated API calls into a PhaseConfig object.
    Returns the populated PhaseConfig, or None if execution fails.
    """
    config = PhaseConfig()

    def set_ee_target(target, z_offset=0.0):
        valid = ("box", "handoff_zone", "goal_zone")
        if target not in valid:
            print(f"[reward_function] Unknown target '{target}', defaulting to 'box'")
            config.target = "box"
        else:
            config.target = target
        config.z_offset = float(z_offset)

    def set_done_threshold(distance):
        config.done_threshold = float(distance)

    def set_alignment_weight(weight):
        config.alignment_weight = float(weight)

    def set_alignment_threshold(min_alignment):
        config.alignment_threshold = float(min_alignment)

    def set_gripper(state):
        valid = ("open", "closed", "open_when_done")
        if state not in valid:
            print(f"[reward_function] Unknown gripper state '{state}', defaulting to 'open'")
            config.gripper_state = "open"
        else:
            config.gripper_state = state

    code = text.strip()
    code = re.sub(r"^```python\s*", "", code)
    code = re.sub(r"\s*```$",       "", code)

    try:
        exec(code, {
            "set_ee_target":           set_ee_target,
            "set_done_threshold":      set_done_threshold,
            "set_alignment_weight":    set_alignment_weight,
            "set_alignment_threshold": set_alignment_threshold,
            "set_gripper":             set_gripper,
        })
    except Exception as e:
        print(f"[reward_function] API call parse failed for phase {phase.phase_id}: {e}")
        return None

    return config


# -------------------------
# FALLBACK
# -------------------------

def _fallback_config(phase):
    """Returns a sensible PhaseConfig based on action type when Gemini is unavailable."""
    config = PhaseConfig()

    if phase.action == "reach":
        config.target              = "box"
        config.z_offset            = 0.0
        config.gripper_state       = "open"
        config.alignment_weight    = 0.5
        config.alignment_threshold = 0.5

    elif phase.action == "grasp":
        config.target              = "box"
        config.z_offset            = 0.0
        config.gripper_state       = "closed"
        config.alignment_weight    = 0.5
        config.alignment_threshold = 0.4

    elif phase.action == "place":
        config.target              = "handoff_zone" if phase.arm == "A" else "goal_zone"
        config.z_offset            = 0.05
        config.gripper_state       = "open_when_done"
        config.alignment_weight    = 0.0
        config.alignment_threshold = -1.0

    elif phase.action == "clear_obstacle":
        config.target              = "box"
        config.z_offset            = 0.10
        config.gripper_state       = "open"
        config.alignment_weight    = 0.0
        config.alignment_threshold = -1.0

    return config


# -------------------------
# PHASE FUNCTIONS
# -------------------------

class PhaseFunctions:
    def __init__(self, phase, reward_fn, is_done_fn, gripper_fn, source):
        self.phase      = phase
        self.source     = source
        self.reward_fn  = reward_fn
        self.is_done_fn = is_done_fn
        self.gripper_fn = gripper_fn

    def reward(self, data, site_a_id, site_b_id, object_body_id,
               obstacle_body_id, handoff_pos, goal_pos):
        return self.reward_fn(data, site_a_id, site_b_id, object_body_id,
                              obstacle_body_id, handoff_pos, goal_pos)

    def is_done(self, data, site_a_id, site_b_id, object_body_id,
                obstacle_body_id, handoff_pos, goal_pos):
        return bool(self.is_done_fn(data, site_a_id, site_b_id, object_body_id,
                                    obstacle_body_id, handoff_pos, goal_pos))

    def gripper(self, data, site_a_id, site_b_id, object_body_id,
                obstacle_body_id, handoff_pos, goal_pos):
        return float(self.gripper_fn(data, site_a_id, site_b_id, object_body_id,
                                     obstacle_body_id, handoff_pos, goal_pos))


def build_phase_functions(phase, scene_state: dict):
    config = _call_gemini(phase, scene_state)
    if config is not None:
        fns = _build_functions(config, phase, scene_state)
        print(f"[reward_function] Phase {phase.phase_id}: config from Gemini.")
        return PhaseFunctions(phase, *fns, source="gemini")

    print(f"[reward_function] Phase {phase.phase_id}: using fallback config.")
    config = _fallback_config(phase)
    fns    = _build_functions(config, phase, scene_state)
    return PhaseFunctions(phase, *fns, source="fallback")


# -------------------------
# DECOMPOSITION
# -------------------------

def decompose_phase(phase, reason="timeout"):
    """
    Asks Gemini to split a failed original phase into 2-3 sub-phases.
    Only called for original phases (phase_id < 100) to prevent recursive decomposition.
    """
    if phase.phase_id >= 100:
        print(f"[reward_function] Phase {phase.phase_id} is a sub-phase, skipping decompose.")
        return None

    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        client = genai.Client(api_key=api_key)
        prompt = f"""
Phase {phase.phase_id} (Arm {phase.arm}, action='{phase.action}', target='{phase.target}')
failed due to: {reason}.

Decompose into 2-3 smaller sequential sub-phases achieving the same goal.
Use ONLY these valid target values: box, handoff_zone, center, A, B.
Use ONLY these valid action values: reach, grasp, place, clear_obstacle.

Output ONLY a JSON array:
[
  {{"arm": "A or B", "action": "reach|grasp|place|clear_obstacle", "target": "box|handoff_zone|center|A|B"}},
  ...
]
No explanation, no markdown fences.
"""

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.1),
        )

        print(f"\n[gemini decompose]\n{response.text}\n")

        text = response.text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$",     "", text)

        raw        = json.loads(text)
        sub_phases = []
        for i, sp in enumerate(raw):
            sub_phases.append(Phase(
                phase_id=phase.phase_id * 100 + i,
                arm=sp["arm"],
                action=sp["action"],
                target=sp["target"],
                depends_on=None,
            ))

        print(f"[reward_function] Phase {phase.phase_id} decomposed into {len(sub_phases)} sub-phases.")
        return sub_phases

    except Exception as e:
        print(f"[reward_function] Decompose failed for phase {phase.phase_id}: {e}")
        return None


if __name__ == "__main__":
    from motion_descriptor import parse_task

    task = parse_task("move box from table A to table B")
    dummy_scene = {
        "box_pos":     np.array([-0.96, 0.0, 0.035]),
        "handoff_pos": np.array([0.0,   0.0, 0.035]),
        "goal_pos":    np.array([0.96,  0.0, 0.035]),
        "ee_a":        np.array([-0.14, -0.005, 0.234]),
        "ee_b":        np.array([ 0.14, -0.033, 0.234]),
        "ee_a_xmat":   np.eye(3),
        "ee_b_xmat":   np.eye(3),
        "min_ee_z":    0.05,
    }
    for phase in task.phases:
        pf = build_phase_functions(phase, dummy_scene)
        print(f"  Phase {phase.phase_id}: source={pf.source}")
        time.sleep(2)
