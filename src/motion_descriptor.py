"""
Rule-based motion descriptor.

Takes a plain-text task string and returns a structured task decomposition
describing which arm handles which phase, what the target objects are,
and what the handoff zone is.

For now this is intentionally rule-based and simple. 
The output format mirrors the concept from Language to Rewards
(Yu et al., 2023) where motion descriptor produce structured motion
descriptions that feed into a reward generator.
"""

from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class Phase:
    phase_id: int
    arm: str           # "A" (left) or "B" (right)
    action: str        # "reach", "grasp", "place", "clear_obstacle"
    target: str        # object name or zone name
    depends_on: Optional[int] = None  # phase_id this phase waits for


@dataclass
class TaskDecomposition:
    task_str: str
    object_name: str
    obstacle_name: Optional[str]
    start_zone: str    # "A", "B", or "center"
    goal_zone: str
    handoff_required: bool
    phases: list = field(default_factory=list)

    def describe(self):
        lines = [
            f"Task       : {self.task_str}",
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


def _has_obstacle_action(text):
    return any(kw in text for kw in _OBSTACLE_ACTIONS)


def _arm_for_zone(zone):
    return "A" if zone == "A" else "B"


def parse_task(task_str):
    """
    Parses a plain-text task string into a TaskDecomposition.

    Supported patterns:
        "move [object] from [zone] to [zone]"
        "move [object] behind [obstacle] from [zone] to [zone]"
        "pick [object] on [zone] and place on [zone]"

    Examples:
        parse_task("move box from table A to table B")
        parse_task("move box behind obstacle from table A to table B")
    """
    text = task_str.lower().strip()

    object_name   = _extract_object(text)
    obstacle_name = _extract_obstacle(text) if _has_obstacle_action(text) else None

    from_match = re.search(r"from (.+?) to (.+?)(?:\s|$)", text)
    on_match   = re.search(r"on (.+?) (?:and )?(?:place )?(?:on|to) (.+?)(?:\s|$)", text)

    start_zone = goal_zone = None

    if from_match:
        start_zone = _extract_zone(from_match.group(1))
        goal_zone  = _extract_zone(from_match.group(2))
    elif on_match:
        start_zone = _extract_zone(on_match.group(1))
        goal_zone  = _extract_zone(on_match.group(2))

    if start_zone is None or goal_zone is None:
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

    start_arm = _arm_for_zone(start_zone)
    goal_arm  = _arm_for_zone(goal_zone)
    handoff   = start_arm != goal_arm

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
    )


if __name__ == "__main__":
    examples = [
        "move box from table A to table B",
        "move box behind obstacle from table A to table B",
        "pick cube on left table and place on right table",
        "move eraser from right side to left side",
    ]
    for task in examples:
        print("\n" + "=" * 55)
        print(parse_task(task).describe())
    print("=" * 55)
