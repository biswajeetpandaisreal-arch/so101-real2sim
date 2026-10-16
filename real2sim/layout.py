"""The real table layout and its mapping into the simulator frame.

Positions on the real table are measured with a tape measure from the
shoulder-pan axis (the centre of the rotating base): `fwd_cm` straight out in
front of the robot, `left_cm` to the robot's left (negative = right). The
MuJoCo model's origin is the base frame, 38.8 mm behind the pan axis.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

LAYOUT = Path(__file__).resolve().parent.parent / "config" / "layout.csv"
PAN_AXIS_X = 0.0388            # pan axis in the MuJoCo base frame (m), from so101.xml


def to_sim(fwd_cm: float, left_cm: float) -> np.ndarray:
    return np.array([fwd_cm / 100.0 + PAN_AXIS_X, left_cm / 100.0])


def read_layout(path: Path = LAYOUT) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["name"]: {"role": r["role"], "xy": to_sim(float(r["fwd_cm"]), float(r["left_cm"]))}
                for r in csv.DictReader(f)}


def train_hull(layout: dict[str, dict]):
    """Convex hull of the demonstrated cube positions (scipy Delaunay)."""
    from scipy.spatial import Delaunay
    return Delaunay(np.array([s["xy"] for s in layout.values() if s["role"] == "train"]))


def is_interpolation(xy, hull) -> bool:
    return bool(hull.find_simplex(np.asarray(xy)[None])[0] >= 0)
