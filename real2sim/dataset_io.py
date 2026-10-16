"""LeRobot datasets in the SO-101 recording format.

Simulated datasets use the feature names `lerobot-record` produces for an SO-101
follower (joint positions in degrees, gripper 0-100), so real and simulated
episodes are interchangeable for replay and training.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .sim_env import FPS, JOINTS

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
