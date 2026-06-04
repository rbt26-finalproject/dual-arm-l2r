"""
types_.py

Shared dataclasses used across the dual-arm manipulation pipeline.
Kept in a separate module to avoid circular imports between
motion_descriptor.py, reward_function.py, and mujoco_runner.py.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Phase:
    phase_id: int
    arm: str
    action: str
    target: str
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
