"""Recordings and datasets: LeRobot I/O, what each demonstration says about
itself, and replaying it in the twin.

Simulated datasets use the feature names `lerobot-record` produces for an SO-101
follower (joint positions in degrees, gripper 0-100), so real and simulated
episodes are interchangeable. The real object's position in each episode is
recovered from the arm: when the real jaws stall on the object (gripper
commanded shut, follower gripper held open by the object), forward kinematics
puts the grasp centre on the object. The frames covering exactly one
pick-and-place are located the same way, so waiting time and half-done second
attempts are cut.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np

from .kinematics import SO101IK
from .sim import FPS, JOINTS

MOTOR_NAMES = [f"{j}.pos" for j in JOINTS]
# Task text stored with every frame. The recordings carry the wording of the first
# (cube) setup; ACT does not read it.
TASK = "Pick up the red cube and place it in the blue basket."


def sim_features(cameras: dict[str, tuple[int, int]]) -> dict:
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": MOTOR_NAMES},
             "action": {"dtype": "float32", "shape": (6,), "names": MOTOR_NAMES}}
    for name, (h, w) in cameras.items():
        feats[f"observation.images.{name}"] = {"dtype": "video", "shape": (h, w, 3),
                                               "names": ["height", "width", "channels"]}
    return feats


class SimDatasetWriter:
    """Writes simulated episodes (rendered cameras, joint state and command)."""

    def __init__(self, root: Path, repo_id: str, cameras=("top", "wrist_cam"), size=(240, 320)):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root = Path(root)
        if root.exists():
            shutil.rmtree(root)
        self.cameras = list(cameras)
        # Streaming encoding encodes frames as they arrive; the default (PNG files,
        # then encode) is ~10x slower and dominated dataset build time.
        self.ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, root=root, robot_type="so101_follower",
                                        features=sim_features({c: size for c in self.cameras}),
                                        use_videos=True, streaming_encoding=True)

    def add(self, env, action):
        frame = {"observation.state": env.state_lerobot().astype(np.float32),
                 "action": np.asarray(action, np.float32), "task": TASK}
        for c in self.cameras:
            frame[f"observation.images.{c}"] = env.render(c)
        self.ds.add_frame(frame)

    def end_episode(self, keep: bool = True):
        if keep:
            self.ds.save_episode()
        else:
            self.ds.clear_episode_buffer()

    def close(self):
        self.ds.finalize()


def load_episodes(root: Path, repo_id: str = "local/so101_real") -> list[dict]:
    """Joint trajectories of every episode: dicts with 'episode' and 'action' /
    'state' arrays (T x 6, LeRobot units). Videos are not decoded."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")      # torchcodec does not load on Windows
    cols = ds.hf_dataset.with_format("numpy")
    ep = np.asarray(cols["episode_index"])
    act = np.asarray(cols["action"], np.float64)
    state = np.asarray(cols["observation.state"], np.float64)
    return [{"episode": int(e), "action": act[ep == e], "state": state[ep == e]} for e in np.unique(ep)]


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
