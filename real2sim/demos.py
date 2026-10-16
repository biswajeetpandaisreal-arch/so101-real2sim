"""Recorded demonstrations: what each one says about itself, and replaying it in the twin.

The real object's position in each episode is recovered from the arm: when the
real jaws stall on the object (gripper commanded shut, follower gripper held
open by the object), forward kinematics puts the grasp centre on the object.
The frames covering exactly one pick-and-place are located the same way, so
waiting time and half-done second attempts are cut.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .dataset_io import load_episodes
from .kinematics import SO101IK


def grasp_frame(action, state, closed_cmd=10.0, blocked=6.0, hold=10) -> int | None:
    """First frame of a sustained stall: command near shut while the jaw is held at
    least `blocked` open for `hold` frames. None if the jaws never closed on anything."""
    stall = (action[:, 5] < closed_cmd) & (state[:, 5] - action[:, 5] > blocked)
    return next((k for k in range(len(stall) - hold) if stall[k:k + hold].all()), None)


def object_position(env, ik: SO101IK, action, state):
    """((x, y) of the grasped object in the sim frame, stall frame), or (None, None)."""
    k = grasp_frame(action, state)
    if k is None:
        return None, None
    return ik.grasp_point(env.lerobot_to_rad(state[k])[:5])[:2], k


def active_span(state, tol=5.0, pad=15):
    """(first, last) frame of the operator's motion, padded. Motion starts when
    shoulder_lift or elbow_flex leaves its starting angle by more than tol deg, so
    wrist twitches and gripper settling at rest do not count."""
    big = np.abs(state[:, 1:3] - state[0, 1:3]).max(1) > tol
    if not big.any():
        return None
    start = int(np.argmax(big))
    moving = np.where(np.abs(np.diff(state, axis=0)).sum(1) > 0.5)[0]
    end = int(moving[-1]) + 1 if len(moving) else len(state)
    return max(0, start - pad), min(len(state), max(end, start) + pad)


def task_span(action, state, open_cmd=30.0, home_tol=15.0, pad=15):
    """(first, last) frame of exactly one pick-and-place: from the start of motion
    to the return near the starting pose after the first release. None without a grasp."""
    span, k = active_span(state, pad=pad), grasp_frame(action, state)
    if span is None or k is None:
        return None
    a0, a1 = span
    opened = np.where(action[k:, 5] > open_cmd)[0]
    if not len(opened):
        return a0, a1                                             # never released: keep it all
    r = k + int(opened[0])
    home = np.where(np.abs(state[r:, :5] - state[a0, :5]).max(1) < home_tol)[0]
    return a0, min(r + int(home[0]) + pad if len(home) else a1, a1)


def read_episode_list(path: Path) -> dict[int, dict]:
    """episodes.csv (scripts/make_episode_list.py): episode -> mark, object pose, task frames."""
    with open(path, newline="") as f:
        return {int(r["episode"]): {"mark": r["mark"], "xy": np.array([float(r["x"]), float(r["y"])]),
                                    "yaw": np.radians(float(r["yaw_deg"])),
                                    "frames": (int(r["start"]), int(r["end"]))}
                for r in csv.DictReader(f)}


def load_demos(data: Path, repo_id: str, episodes: dict[int, dict]) -> list[dict]:
    """The listed episodes, trimmed to their task frames, each with its list entry under 'info'."""
    out = []
    for ep in load_episodes(data, repo_id):
        if (info := episodes.get(ep["episode"])) is not None:
            a, b = info["frames"]
            out.append({**ep, "action": ep["action"][a:b], "state": ep["state"][a:b], "info": info})
    return out


def replay(env, demo, writer=None) -> bool:
    """Drive the twin with a demonstration's recorded leader commands; True if
    the object ends in the box. Frames are written to `writer` if given."""
    env.reset(demo["info"]["xy"], demo["info"]["yaw"], q0_lerobot=demo["state"][0])
    for a in demo["action"]:
        if writer is not None:
            writer.add(env, a)
        env.step(a)
    return env.success()
