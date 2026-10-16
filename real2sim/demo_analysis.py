"""What a recorded demonstration says about itself.

The real object's position in each episode is recovered from the arm: the
moment the real jaws stall on the object (gripper commanded shut, follower
gripper stopped well short of it), forward kinematics puts the grasp centre on
the object. Idle time before and after the operator moves is also located, so
it can be trimmed.
"""

from __future__ import annotations

import numpy as np

from .kinematics import SO101IK


def grasp_frame(action, state, closed_cmd=10.0, blocked=6.0, hold=10):
    """First frame of a sustained stall: command near shut, jaw held open by the
    object for `hold` consecutive frames. None if the jaws never closed on anything."""
    stall = (action[:, 5] < closed_cmd) & (state[:, 5] - action[:, 5] > blocked)
    for k in range(len(stall) - hold):
        if stall[k:k + hold].all():
            return k
    return None


def object_position(env, ik: SO101IK, action, state):
    """(x, y) of the grasped object in the sim frame, from the stall frame."""
    k = grasp_frame(action, state)
    if k is None:
        return None, None
    p = ik.grasp_point(env.lerobot_to_rad(state[k])[:5])
    return p[:2], k


def active_span(state, tol=5.0, pad=15):
    """Frames from when the arm sets off to when it last moves, padded. The arm
    "sets off" when shoulder_lift or elbow_flex leaves its starting angle by
    more than tol degrees (wrist twitches and gripper settling at rest do not
    count, so waiting time before the demo is cut)."""
    big = np.abs(state[:, 1:3] - state[0, 1:3]).max(1) > tol
    if not big.any():
        return None
    start = int(np.argmax(big))
    moving = np.where(np.abs(np.diff(state, axis=0)).sum(1) > 0.5)[0]
    end = int(moving[-1]) + 1 if len(moving) else len(state)
    return max(0, start - pad), min(len(state), max(end, start) + pad)


def executed(action, state):
    """The trajectory the real arm actually executed: the follower's measured
    arm joints, with the operator's gripper command (the measured gripper stops
    on the object, so commanding it would not squeeze). The real STS3215 servos
    (P gain 16) lag and smooth the leader's motion by up to ~40 deg on fast
    moves; the simulated actuators do not, so replaying the raw leader command
    whips the object out of the jaws."""
    return np.c_[state[:, :5], action[:, 5:]]


def task_span(action, state, open_cmd=30.0, home_tol=15.0, pad=15):
    """Frames covering exactly one pick-and-place: from when the arm starts
    moving to when it is back near its starting pose after the first release
    (an operator who keeps going would otherwise add a second, half-done pick
    to the same episode). None if there is no grasp."""
    span = active_span(state, pad=pad)
    k = grasp_frame(action, state)
    if span is None or k is None:
        return None
    a0, a1 = span
    opened = np.where(action[k:, 5] > open_cmd)[0]
    if not len(opened):
        return a0, a1                                       # never released: keep everything
    r = k + int(opened[0])
    near_home = np.where(np.abs(state[r:, :5] - state[a0, :5]).max(1) < home_tol)[0]
    end = r + int(near_home[0]) + pad if len(near_home) else a1
    return a0, min(end, a1)
