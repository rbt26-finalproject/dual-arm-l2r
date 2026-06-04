"""
motion_descriptor.py

Two-stage task parser following Language to Rewards (Yu et al., 2023).

Stage 1 (Motion Descriptor): Gemini converts a natural language task string
into a structured motion plan using a constrained template. This separates
task understanding from reward coding.

Stage 2 (Task Decomposition): The structured plan is parsed into a
TaskDecomposition with ordered Phase objects consumed by the runner.

Falls back to rule-based parsing when GEMINI_API_KEY is not set or the
API call fails.
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional
from google import genai
from google.genai import types


@dataclass
class Phase:
    phase_id: int
    arm: str           # "A" (left) or "B" (right)
    action: str        # "reach", "grasp", "place", "clear_obstacle"
    target: str        # object name or zone name
    depends_on: Optional[int] = None


@dataclass
class TaskDecomposition:
    task_str: str
    object_name: str
    obstacle_name: Optional[str]
    start_zone: str
    goal_zone: str
    handoff_required: bool
    phases: list = field(default_factory=list)
    source: str = "unknown"

    def describe(self):
        lines = [
            f"Task       : {self.task_str}",
            f"Source     : {self.source}",
            f"Object     : {self.object_name}",
            f"Obstacle   : {self.obstacle_name or 'none'}",
            f"Start zone : {self.start_zone}",
            f"Goal zone  : {self.goal_zone}",
            f"Handoff    : {'yes' if self.handoff_required else 'no'}",
            "Phases     :",
        ]
        for p in self.phases:
            dep = f" (after phase {p.depends_on})" if p.depends_on is not None else ""
            lines.append(f"  Phase {p.phase_id}: Arm {p.arm} -> {p.action} [{p.target}]{dep}")
        return "\n".join(lines)


# -------------------------
# STAGE 1: MOTION DESCRIPTOR
# -------------------------

MOTION_DESCRIPTOR_SYSTEM_PROMPT = """
You are a motion planner for a dual-arm robot manipulation system.

The scene contains: box, obstacle (optional).
The arms are:
  Arm A (left):  can reach table_A and center table only.
  Arm B (right): can reach table_B and center table only.
  Center table:  reachable by both arms, used as handoff zone.

Given a task, output a structured motion plan using ONLY this template:

[start of plan]
Object to manipulate: {CHOICE: box}
Start location: {CHOICE: table_A, table_B, center}
Goal location: {CHOICE: table_A, table_B, center}
Handoff required: {CHOICE: yes, no}
[optional] Obstacle to clear: {CHOICE: obstacle}
[optional] Arm A clears obstacle before reaching object.
Arm {CHOICE: A, B} reaches the object.
Arm {CHOICE: A, B} grasps the object.
[optional] Arm {CHOICE: A, B} places the object at center for handoff.
[optional] Arm {CHOICE: A, B} reaches the handoff zone.
[optional] Arm {CHOICE: A, B} grasps the object from handoff zone.
Arm {CHOICE: A, B} places the object at goal location.
[end of plan]

Rules:
1. Replace {CHOICE: ...} with exactly one of the listed options.
2. Only include [optional] lines when necessary.
3. If object starts at table_A and goal is table_B (or vice versa), handoff is required.
4. Output ONLY the plan between [start of plan] and [end of plan], no explanation.
"""


def _call_motion_descriptor(task_str):
    """Stage 1: converts natural language task to structured motion plan string."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=task_str,
            config=types.GenerateContentConfig(
                system_instruction=MOTION_DESCRIPTOR_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )

        text = response.text.strip()
        print(f"[motion_descriptor] Plan:\n{text}\n")
        return text

    except Exception as e:
        print(f"[motion_descriptor] Stage 1 failed: {e}")
        return None


# -------------------------
# STAGE 2: PLAN -> STRUCTURED DECOMPOSITION
# -------------------------

DECOMPOSER_SYSTEM_PROMPT = """
You are a task decomposer for a dual-arm robot.

Given a structured motion plan, output ONLY a JSON object:
{
  "object_name": "string",
  "obstacle_name": "string or null",
  "start_zone": "A or B or center",
  "goal_zone": "A or B or center",
  "handoff_required": true or false,
  "phases": [
    {
      "phase_id": 1,
      "arm": "A or B",
      "action": "clear_obstacle or reach or grasp or place",
      "target": "box or obstacle or handoff_zone or A or B or center or above_box",
      "depends_on": null or integer
    }
  ]
}

Map plan lines to phases strictly in order. No explanation. No markdown fences.
"""


def _call_decomposer(plan_text):
    """Stage 2: converts structured plan text to JSON task decomposition."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=plan_text,
            config=types.GenerateContentConfig(
                system_instruction=DECOMPOSER_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )

        raw = response.text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$",     "", raw)

        data = json.loads(raw)

        phases = [
            Phase(
                phase_id=p["phase_id"],
                arm=p["arm"],
                action=p["action"],
                target=p["target"],
                depends_on=p.get("depends_on"),
            )
            for p in data["phases"]
        ]

        return TaskDecomposition(
            task_str="",
            object_name=data["object_name"],
            obstacle_name=data.get("obstacle_name"),
            start_zone=data["start_zone"],
            goal_zone=data["goal_zone"],
            handoff_required=data["handoff_required"],
            phases=phases,
            source="gemini",
        )

    except Exception as e:
        print(f"[motion_descriptor] Stage 2 failed: {e}")
        return None


# -------------------------
# RULE-BASED FALLBACK
# -------------------------

_ZONE_KEYWORDS = {
    "table a": "A", "left table": "A", "left side": "A",
    "table b": "B", "right table": "B", "right side": "B",
    "center": "center", "middle": "center",
}
_OBJECT_KEYWORDS   = ["box", "cube", "eraser", "object", "item", "block"]
_OBSTACLE_KEYWORDS = ["obstacle", "cup", "bottle", "blocker"]
_OBSTACLE_ACTIONS  = ["behind", "blocked", "blocking", "clear", "move aside"]


def _extract_zone(text):
    for kw, zone in _ZONE_KEYWORDS.items():
        if kw in text:
            return zone
    return None


def _extract_object(text):
    for kw in _OBJECT_KEYWORDS:
        if kw in text:
            return kw
    return "object"


def _extract_obstacle(text):
    for kw in _OBSTACLE_KEYWORDS:
        if kw in text:
            return kw
    return None


def _rule_based_parse(task_str):
    text = task_str.lower().strip()

    object_name   = _extract_object(text)
    obstacle_name = _extract_obstacle(text) if any(k in text for k in _OBSTACLE_ACTIONS) else None

    from_match = re.search(r"from (.+?) to (.+?)$", text)
    on_match   = re.search(r"on (.+?) (?:and )?(?:place )?(?:on|to) (.+?)$", text)

    start_zone = goal_zone = None
    if from_match:
        start_zone = _extract_zone(from_match.group(1))
        goal_zone  = _extract_zone(from_match.group(2))
    elif on_match:
        start_zone = _extract_zone(on_match.group(1))
        goal_zone  = _extract_zone(on_match.group(2))

    if not start_zone or not goal_zone:
        found = []
        for kw, zone in _ZONE_KEYWORDS.items():
            if kw in text and zone not in found:
                found.append(zone)
        if len(found) >= 2:
            start_zone, goal_zone = found[0], found[1]
        elif len(found) == 1:
            start_zone = found[0]
            goal_zone  = "B" if start_zone == "A" else "A"
        else:
            start_zone, goal_zone = "A", "B"

    _zone_to_arm = {"A": "A", "B": "B", "center": None}
    start_arm = _zone_to_arm.get(start_zone, "A")
    goal_arm  = _zone_to_arm.get(goal_zone,  "B")

    if start_arm is None:
        start_arm = "B" if goal_arm == "A" else "A"
    if goal_arm is None:
        goal_arm  = "B" if start_arm == "A" else "A"

    handoff  = start_arm != goal_arm
    phases   = []
    phase_id = 1

    if obstacle_name:
        phases.append(Phase(phase_id=phase_id, arm=start_arm,
                            action="clear_obstacle", target=obstacle_name))
        phase_id += 1

    phases.append(Phase(phase_id=phase_id, arm=start_arm, action="reach",
                        target=object_name,
                        depends_on=phase_id - 1 if obstacle_name else None))
    phase_id += 1

    phases.append(Phase(phase_id=phase_id, arm=start_arm, action="grasp",
                        target=object_name, depends_on=phase_id - 1))
    phase_id += 1

    if handoff:
        phases.append(Phase(phase_id=phase_id, arm=start_arm, action="place",
                            target="handoff_zone", depends_on=phase_id - 1))
        phase_id += 1
        phases.append(Phase(phase_id=phase_id, arm=goal_arm, action="reach",
                            target="handoff_zone", depends_on=phase_id - 1))
        phase_id += 1
        phases.append(Phase(phase_id=phase_id, arm=goal_arm, action="grasp",
                            target=object_name, depends_on=phase_id - 1))
        phase_id += 1

    phases.append(Phase(phase_id=phase_id, arm=goal_arm, action="place",
                        target=goal_zone, depends_on=phase_id - 1))

    return TaskDecomposition(
        task_str=task_str,
        object_name=object_name,
        obstacle_name=obstacle_name,
        start_zone=start_zone,
        goal_zone=goal_zone,
        handoff_required=handoff,
        phases=phases,
        source="rule_based",
    )


# -------------------------
# PUBLIC ENTRY POINT
# -------------------------

def parse_task(task_str):
    """
    Parses a natural language task string into a TaskDecomposition.

    Follows the two-stage pipeline from Language to Rewards (Yu et al., 2023):
      Stage 1: Motion Descriptor — LLM produces a constrained plan template.
      Stage 2: Decomposer — LLM converts plan to structured JSON phases.

    Falls back to rule-based parsing if either stage fails.
    """
    plan_text = _call_motion_descriptor(task_str)
    if plan_text:
        result = _call_decomposer(plan_text)
        if result is not None:
            result.task_str = task_str
            return result

    print("[motion_descriptor] Falling back to rule-based parser.")
    return _rule_based_parse(task_str)


if __name__ == "__main__":
    import time

    examples = [
        "move the box from table A to table B",
        "pick up the eraser that is behind the cup on the left table and place it on the right table",
        "take the cube on the right side and move it to the left table",
        "move box behind obstacle from table A to table B",
    ]
    for i, task in enumerate(examples):
        print("=" * 60)
        print(parse_task(task).describe())
        if i < len(examples) - 1:
            time.sleep(2)
    print("=" * 60)
