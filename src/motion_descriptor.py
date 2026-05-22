"""
Motion descriptor for the dual-arm task.

Calls Gemini to parse a natural language task string into a structured
TaskDecomposition, following the Motion Descriptor concept from Language
to Rewards (Yu et al., 2023). Falls back to a rule-based parser when
GEMINI_API_KEY is not set or the API call fails.

Usage:
    export GEMINI_API_KEY=your_key_here
    python src/motion_descriptor.py
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
    source: str = "unknown"  # "gemini" or "rule_based"

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


GEMINI_SYSTEM_PROMPT = """
You are a robot task planner for a dual-arm manipulation system.

The scene has:
- Arm A: left arm, can only reach table A (left side table) and the center table.
- Arm B: right arm, can only reach table B (right side table) and the center table.
- Center table: reachable by both arms, used as handoff zone.
- table A: left side, only Arm A can reach it.
- table B: right side, only Arm B can reach it.

Given a task string, output ONLY a JSON object with this exact structure:
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
      "target": "object name or zone name",
      "depends_on": null or phase_id integer
    }
  ]
}

Rules:
- If object starts on table A and goal is on table B, handoff is required.
- Handoff is placed at center table, picked up by Arm B.
- If there is an obstacle, Arm A must clear it before reaching the object.
- Output ONLY the JSON, no explanation, no markdown fences.
"""


def _call_gemini(task_str):
    """Calls Gemini API and returns parsed TaskDecomposition or None on failure."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            return None

        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=task_str,
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )

        raw = response.text.strip()

        # Strip markdown fences if the model added them despite instructions
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

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
            task_str=task_str,
            object_name=data["object_name"],
            obstacle_name=data.get("obstacle_name"),
            start_zone=data["start_zone"],
            goal_zone=data["goal_zone"],
            handoff_required=data["handoff_required"],
            phases=phases,
            source="gemini",
        )

    except Exception as e:
        print(f"[motion_descriptor] Gemini call failed: {e}. Falling back to rule-based.")
        return None


# Rule-based fallback

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

    handoff = start_arm != goal_arm

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


def parse_task(task_str):
    """
    Parses a natural language task string into a TaskDecomposition.
    Uses Gemini if GEMINI_API_KEY is set, otherwise falls back to rule-based.
    """
    result = _call_gemini(task_str)
    if result is not None:
        return result
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
            time.sleep(2)  # To not trigger TooManyReuqests.
    print("=" * 60)
