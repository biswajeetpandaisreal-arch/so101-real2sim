"""MuJoCo twin of the SO-101 pick-and-place setup.

Commands use LeRobot units (degrees for the five arm joints, 0-100 for the
gripper), so recorded teleoperation drives the simulated arm directly. Three
measured quantities make the twin match the real arm, each applied when its
file exists:

  config/calibration/so101_follower.json   joint limits (the real arm rests at
                                           wrist_flex -102 deg, beyond the
                                           Menagerie limit of +/-95 deg)
  sim/actuators_fitted.json                identified servo dynamics
                                           (scripts/sysid_fit.py)
  config/layout.csv                        where the box is on the table
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .layout import read_layout

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "sim" / "scene_task.xml"
ACTUATORS = ROOT / "sim" / "actuators_fitted.json"
CALIBRATION = ROOT / "config" / "calibration" / "so101_follower.json"
LAYOUT = ROOT / "config" / "layout.csv"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30                                            # LeRobot recording rate
ACTUATOR_PARAMS = ("kp", "kv", "damping", "frictionloss", "armature")


def _load(path_or_value):
    if isinstance(path_or_value, Path):
        return json.loads(path_or_value.read_text()) if path_or_value.exists() else None
    return path_or_value


class SO101Env:
    def __init__(self, render_size: tuple[int, int] = (240, 320), actuators: Path | dict | None = ACTUATORS,
                 calibration: Path | dict | None = CALIBRATION, layout: Path | None = LAYOUT):
        m = self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(m)
        self.jnt = [m.joint(n).id for n in JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jnt])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.jnt])
        self.act = np.array([m.actuator(n).id for n in JOINTS])
        if (calib := _load(calibration)) is not None:
            self._set_limits(calib)
        self.range = np.array([m.jnt_range[j] for j in self.jnt])
        if layout is not None and layout.exists() and "box" in (marks := read_layout(layout)):
            m.body("box").pos[:2] = marks["box"]["xy"]
        self.box_pos = m.body("box").pos.copy()
        self.box_half = m.geom("box_floor").size[:2] - 0.005          # usable floor half-sizes (m)
        self.box_top = 2 * m.geom("box_wall").size[2] + 0.006
        self.obj_body = m.body("object").id
        self.obj_qadr = m.jnt_qposadr[m.body("object").jntadr[0]]
        self.obj_half_height = m.geom("object").size[1]               # cylinder: (radius, half-height)
        self.substeps = max(1, round(1.0 / FPS / m.opt.timestep))
        self.render_size = render_size
        self._renderer = None
        if (params := _load(actuators)) is not None:
            self.set_actuators(params)

    def _set_limits(self, calib: dict):
        """Arm joint limits from the real arm's calibrated range. LeRobot maps the
        range midpoint to 0 deg, so the range is symmetric; it is never narrowed
        below the model's. The gripper keeps its 0-100 mapping onto the jaw range."""
        m = self.model
        for name, j, a in zip(JOINTS[:5], self.jnt[:5], self.act[:5]):
            c = calib[name]
            half = max(np.deg2rad((c["range_max"] - c["range_min"]) * 360 / 4095 / 2), m.jnt_range[j, 1])
            m.jnt_range[j] = m.actuator_ctrlrange[a] = (-half, half)

    # ── actuators ───────────────────────────────────────────────────────────
    def get_actuators(self) -> dict:
        m = self.model
        return {j: {"kp": float(m.actuator_gainprm[a, 0]), "kv": float(-m.actuator_biasprm[a, 2]),
                    "damping": float(m.dof_damping[d]), "frictionloss": float(m.dof_frictionloss[d]),
                    "armature": float(m.dof_armature[d])}
                for j, a, d in zip(JOINTS, self.act, self.dadr)}

    def set_actuators(self, params: dict):
        """Per-joint {kp, kv, damping, frictionloss, armature}; any subset."""
        m = self.model
        for j, a, d in zip(JOINTS, self.act, self.dadr):
            p = params.get(j, {})
            if "kp" in p:
                m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1] = p["kp"], -p["kp"]
            if "kv" in p:
                m.actuator_biasprm[a, 2] = -p["kv"]
            for key, arr in (("damping", m.dof_damping), ("frictionloss", m.dof_frictionloss),
                             ("armature", m.dof_armature)):
                if key in p:
                    arr[d] = p[key]

    # ── units ───────────────────────────────────────────────────────────────
    def lerobot_to_rad(self, a) -> np.ndarray:
        a = np.asarray(a, float)
        q = np.deg2rad(a)
        lo, hi = self.range[5]
        q[..., 5] = lo + a[..., 5] / 100.0 * (hi - lo)
        return np.clip(q, self.range[:, 0], self.range[:, 1])

    def rad_to_lerobot(self, q) -> np.ndarray:
        q = np.asarray(q, float)
        a = np.rad2deg(q)
        lo, hi = self.range[5]
        a[..., 5] = (q[..., 5] - lo) / (hi - lo) * 100.0
        return a

    # ── episodes ────────────────────────────────────────────────────────────
    def reset(self, obj_xy, obj_yaw: float = 0.0, q0_lerobot=None):
        mujoco.mj_resetData(self.model, self.data)
        q = self.obj_qadr
        self.data.qpos[q:q + 3] = [obj_xy[0], obj_xy[1], self.obj_half_height + 0.001]
        self.data.qpos[q + 3:q + 7] = [np.cos(obj_yaw / 2), 0, 0, np.sin(obj_yaw / 2)]
        if q0_lerobot is not None:
            self.data.qpos[self.qadr] = self.data.ctrl[self.act] = self.lerobot_to_rad(q0_lerobot)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(20):                                           # let the object settle
            mujoco.mj_step(self.model, self.data)

    def step(self, action_lerobot):
        """Apply one 30 Hz command (LeRobot units) as joint position targets."""
        self.data.ctrl[self.act] = self.lerobot_to_rad(action_lerobot)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def state_lerobot(self) -> np.ndarray:
        return self.rad_to_lerobot(self.data.qpos[self.qadr])

    def object_pos(self) -> np.ndarray:
        return self.data.xpos[self.obj_body].copy()

    def success(self) -> bool:
        """Object resting on the box floor, below the wall tops."""
        p = self.object_pos()
        d = np.abs(p[:2] - self.box_pos[:2])
        return bool((d < self.box_half).all() and p[2] < self.box_top)

    # ── rendering ───────────────────────────────────────────────────────────
    # Policy observations use the arm's collision geometry and no shadows: ~10 ms
    # per camera pair instead of ~110 ms with the 400k-triangle CAD meshes on this
    # laptop's integrated GPU. Training and evaluation see the same rendering;
    # pretty=True gives the CAD look for figures.
    FAST_GEOM_GROUPS = (1, 1, 0, 1, 1, 0)

    def render(self, camera: str, pretty: bool = False) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.render_size)
            self._fast = mujoco.MjvOption()
            self._fast.geomgroup[:] = self.FAST_GEOM_GROUPS
        shadows = self.model.light_castshadow.copy()
        if not pretty:
            self.model.light_castshadow[:] = 0
        self._renderer.update_scene(self.data, camera=camera, scene_option=None if pretty else self._fast)
        self.model.light_castshadow[:] = shadows
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
