"""MuJoCo pick-and-place environment for the SO-101.

Joint commands use LeRobot's units, so real teleoperation data drives the
simulated arm directly: degrees for the five arm joints and 0-100 for the
gripper (the mapping is the one validated in the so101_digital_twin project).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "sim" / "scene_task.xml"
# Actuator / joint parameters identified from the real arm (scripts/sysid_fit.py).
# Applied whenever the file exists; SO101Env(actuators=None) gives the
# Menagerie defaults for comparison.
ACTUATORS = Path(__file__).resolve().parent.parent / "sim" / "actuators_fitted.json"
# The real follower's calibration. Its recorded range of motion is wider than the
# Menagerie model's joint limits (e.g. wrist_flex +/-101.5 deg vs +/-95 deg; the arm
# rests at -102 deg), so the arm joints' limits are set from it.
CALIBRATION = Path(__file__).resolve().parent.parent / "config" / "calibration" / "so101_follower.json"
LAYOUT = Path(__file__).resolve().parent.parent / "config" / "layout.csv"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30                                    # LeRobot recording rate


class SO101Env:
    def __init__(self, scene: Path = SCENE, render_size: tuple[int, int] = (240, 320),
                 actuators: Path | dict | None = ACTUATORS, calibration: Path | None = CALIBRATION,
                 layout: Path | None = LAYOUT):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.jnt = [self.model.joint(n).id for n in JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jnt])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in self.jnt])
        self.act = np.array([self.model.actuator(n).id for n in JOINTS])
        if calibration is not None and Path(calibration).exists():
            self._measured_limits(json.loads(Path(calibration).read_text()))
        self.range = np.array([self.model.jnt_range[j] for j in self.jnt])
        cube_j = self.model.body("cube").jntadr[0]
        self.cube_qadr = self.model.jnt_qposadr[cube_j]
        self.cube_dadr = self.model.jnt_dofadr[cube_j]
        self.cube_body = self.model.body("cube").id
        if layout is not None and Path(layout).exists():          # basket where it is on the real table
            from .layout import read_layout
            b = read_layout(Path(layout)).get("basket")
            if b is not None:
                self.model.body("basket").pos[:2] = b["xy"]
        self.basket_pos = self.model.body("basket").pos.copy()
        floor = self.model.geom("basket_bottom").size
        self.basket_half = floor[:2] - 0.005                      # usable floor half-sizes (m)
        self.basket_top = 2 * self.model.geom("basket_wall").size[2] + 0.006
        g = self.model.geom("cube")
        self.cube_half_h = g.size[1] if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER else g.size[2]
        self.tcp = self.model.site("gripperframe").id
        self.substeps = max(1, round(1.0 / FPS / self.model.opt.timestep))
        self.render_size = render_size
        self._renderer = None
        if isinstance(actuators, Path):
            actuators = json.loads(actuators.read_text()) if actuators.exists() else None
        self.actuators = actuators
        if actuators:
            self.set_actuators(actuators)

    def _measured_limits(self, calib: dict):
        """Arm joint limits from the real arm's calibrated range (LeRobot maps the
        range midpoint to 0 deg, so it is symmetric). The gripper keeps its own
        0-100 mapping onto the model's jaw range."""
        m = self.model
        for name, j, a in zip(JOINTS[:5], self.jnt[:5], self.act[:5]):
            c = calib[name]
            half = np.deg2rad((c["range_max"] - c["range_min"]) * 360 / 4095 / 2)
            half = max(half, m.jnt_range[j, 1])                 # never narrower than the model
            m.jnt_range[j] = (-half, half)
            m.actuator_ctrlrange[a] = (-half, half)

    # ── actuator model ──────────────────────────────────────────────────────
    def get_actuators(self) -> dict:
        m = self.model
        return {j: {"kp": float(m.actuator_gainprm[a, 0]), "kv": float(-m.actuator_biasprm[a, 2]),
                    "damping": float(m.dof_damping[d]), "frictionloss": float(m.dof_frictionloss[d]),
                    "armature": float(m.dof_armature[d])}
                for j, a, d in zip(JOINTS, self.act, self.dadr)}

    def set_actuators(self, params: dict):
        """Per-joint {kp, kv, damping, frictionloss, armature} (any subset)."""
        m = self.model
        for j, a, d in zip(JOINTS, self.act, self.dadr):
            p = params.get(j, {})
            if "kp" in p:
                m.actuator_gainprm[a, 0] = p["kp"]
                m.actuator_biasprm[a, 1] = -p["kp"]
            if "kv" in p:
                m.actuator_biasprm[a, 2] = -p["kv"]
            for key, arr in (("damping", m.dof_damping), ("frictionloss", m.dof_frictionloss),
                             ("armature", m.dof_armature)):
                if key in p:
                    arr[d] = p[key]

    # ── unit conversion (LeRobot <-> MuJoCo) ────────────────────────────────
    def lerobot_to_rad(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, float)
        q = np.deg2rad(a)
        lo, hi = self.range[5]
        q[..., 5] = lo + a[..., 5] / 100.0 * (hi - lo)
        return np.clip(q, self.range[:, 0], self.range[:, 1])

    def rad_to_lerobot(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        a = np.rad2deg(q)
        lo, hi = self.range[5]
        a[..., 5] = (q[..., 5] - lo) / (hi - lo) * 100.0
        return a

    # ── episode control ─────────────────────────────────────────────────────
    def reset(self, cube_xy, cube_yaw: float = 0.0, q0_lerobot=None):
        mujoco.mj_resetData(self.model, self.data)
        q = self.cube_qadr
        self.data.qpos[q:q + 3] = [cube_xy[0], cube_xy[1], self.cube_half_h + 0.001]
        self.data.qpos[q + 3:q + 7] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
        if q0_lerobot is not None:
            q0 = self.lerobot_to_rad(q0_lerobot)
            self.data.qpos[self.qadr] = q0
            self.data.ctrl[self.act] = q0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(20):                 # let the cube settle
            mujoco.mj_step(self.model, self.data)

    def step(self, action_lerobot):
        """Apply one 30 Hz action (LeRobot units) as position targets."""
        self.data.ctrl[self.act] = self.lerobot_to_rad(action_lerobot)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def state_lerobot(self) -> np.ndarray:
        return self.rad_to_lerobot(self.data.qpos[self.qadr])

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp].copy()

    def success(self) -> bool:
        """Object resting inside the basket footprint, below the wall tops."""
        c = self.cube_pos()
        d = np.abs(c[:2] - self.basket_pos[:2])
        return bool(d[0] < self.basket_half[0] and d[1] < self.basket_half[1] and c[2] < self.basket_top)

    # ── rendering ───────────────────────────────────────────────────────────
    # Policy observations are rendered with the arm's simplified collision
    # geometry and no shadows: ~11 ms per camera pair instead of ~110 ms with the
    # 400k-triangle CAD meshes on this laptop's integrated GPU. Training and
    # evaluation use the same rendering; pretty=True gives the CAD look for figures.
    FAST_GROUPS = (1, 1, 0, 1, 1, 0)

    def render(self, camera: str, pretty: bool = False) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.render_size)
            self._fast = mujoco.MjvOption()
            self._fast.geomgroup[:] = self.FAST_GROUPS
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
