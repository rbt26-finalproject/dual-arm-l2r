"""
Per-phase reward functions for the dual-arm task.

Each function takes the current MuJoCo data and scene info and returns
a scalar reward in [0, 1] for that phase. Higher = better.

Phase 1: Arm A clears the obstacle (minimize distance EE_A to obstacle).
Phase 2: Arm A reaches the object   (minimize distance EE_A to object).
Phase 3: Arm A places at handoff    (minimize distance object to handoff zone).
Phase 4: Arm B reaches handoff zone (minimize distance EE_B to handoff zone).
Phase 5: Arm B grasps object        (same as phase 4, object now at handoff).
Phase 6: Arm B places at goal       (minimize distance object to goal zone).
"""

import numpy as np
import mujoco


def _ee_pos(data, site_id):
    return data.site_xpos[site_id].copy()


def _body_pos(data, body_id):
    return data.xpos[body_id].copy()


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _dist_to_reward(dist, scale=10.0):
    """Converts a distance to a reward in (0, 1] using exponential decay."""
    return float(np.exp(-scale * dist))


def reward_phase1_clear_obstacle(data, site_a_id, obstacle_body_id):
    """
    Phase 1: Arm A moves toward the obstacle to clear it.
    Reward = how close EE_A is to the obstacle center.
    """
    ee_a     = _ee_pos(data, site_a_id)
    obstacle = _body_pos(data, obstacle_body_id)
    dist     = _dist(ee_a, obstacle)
    return _dist_to_reward(dist)


def reward_phase2_reach_object(data, site_a_id, object_body_id):
    """
    Phase 2: Arm A reaches toward the target object.
    Reward = how close EE_A is to the object center.
    """
    ee_a   = _ee_pos(data, site_a_id)
    obj    = _body_pos(data, object_body_id)
    dist   = _dist(ee_a, obj)
    return _dist_to_reward(dist)


def reward_phase3_place_handoff(data, object_body_id, handoff_pos):
    """
    Phase 3: Arm A places the object at the handoff zone.
    Reward = how close the object is to the handoff zone center.
    """
    obj      = _body_pos(data, object_body_id)
    handoff  = np.array(handoff_pos)
    dist     = _dist(obj[:2], handoff[:2])  # only x-y, ignore z
    return _dist_to_reward(dist, scale=8.0)


def reward_phase4_reach_handoff(data, site_b_id, handoff_pos):
    """
    Phase 4: Arm B moves toward the handoff zone to pick up the object.
    Reward = how close EE_B is to the handoff zone center.
    """
    ee_b    = _ee_pos(data, site_b_id)
    handoff = np.array(handoff_pos)
    dist    = _dist(ee_b, handoff)
    return _dist_to_reward(dist)


def reward_phase5_grasp_object(data, site_b_id, object_body_id):
    """
    Phase 5: Arm B grasps the object at the handoff zone.
    Same structure as phase 4 but tracks the object directly.
    """
    ee_b = _ee_pos(data, site_b_id)
    obj  = _body_pos(data, object_body_id)
    dist = _dist(ee_b, obj)
    return _dist_to_reward(dist)


def reward_phase6_place_goal(data, object_body_id, goal_pos):
    """
    Phase 6: Arm B places the object at the goal position.
    Reward = how close the object is to the goal zone center.
    """
    obj  = _body_pos(data, object_body_id)
    goal = np.array(goal_pos)
    dist = _dist(obj[:2], goal[:2])  # only x-y, ignore z
    return _dist_to_reward(dist, scale=8.0)


PHASE_REWARD_FNS = {
    1: reward_phase1_clear_obstacle,
    2: reward_phase2_reach_object,
    3: reward_phase3_place_handoff,
    4: reward_phase4_reach_handoff,
    5: reward_phase5_grasp_object,
    6: reward_phase6_place_goal,
}


def compute_reward(phase_id, data, site_a_id, site_b_id,
                   object_body_id, obstacle_body_id,
                   handoff_pos, goal_pos):
    """
    Dispatches to the correct reward function for the given phase.
    Returns a scalar reward in (0, 1].
    """
    if phase_id == 1:
        return reward_phase1_clear_obstacle(data, site_a_id, obstacle_body_id)
    elif phase_id == 2:
        return reward_phase2_reach_object(data, site_a_id, object_body_id)
    elif phase_id == 3:
        return reward_phase3_place_handoff(data, object_body_id, handoff_pos)
    elif phase_id == 4:
        return reward_phase4_reach_handoff(data, site_b_id, handoff_pos)
    elif phase_id == 5:
        return reward_phase5_grasp_object(data, site_b_id, object_body_id)
    elif phase_id == 6:
        return reward_phase6_place_goal(data, object_body_id, goal_pos)
    else:
        raise ValueError(f"Unknown phase_id: {phase_id}")
