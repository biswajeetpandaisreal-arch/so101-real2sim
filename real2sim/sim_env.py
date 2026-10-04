"""MuJoCo pick-and-place environment for the SO-101.

Joint commands use LeRobot's units, so real teleoperation data drives the
simulated arm directly: degrees for the five arm joints and 0-100 for the
gripper (the mapping is the one validated in the so101_digital_twin project).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "sim" / "scene_task.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30                                    # LeRobot recording rate
BASKET_HALF = 0.07 - 0.005                  # usable half-width of the basket floor (m)


class SO101Env:
    def __init__(self, scene: Path = SCENE, render_size: tuple[int, int] = (240, 320)):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.jnt = [self.model.joint(n).id for n in JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jnt])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in self.jnt])
        self.act = np.array([self.model.actuator(n).id for n in JOINTS])
        self.range = np.array([self.model.jnt_range[j] for j in self.jnt])
        cube_j = self.model.body("cube").jntadr[0]
        self.cube_qadr = self.model.jnt_qposadr[cube_j]
        self.cube_dadr = self.model.jnt_dofadr[cube_j]
        self.cube_body = self.model.body("cube").id
        self.basket_pos = self.model.body("basket").pos.copy()
        self.tcp = self.model.site("gripperframe").id
        self.substeps = max(1, round(1.0 / FPS / self.model.opt.timestep))
        self.render_size = render_size
        self._renderer = None

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
        self.data.qpos[q:q + 3] = [cube_xy[0], cube_xy[1], 0.03]
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
        """Cube resting inside the basket footprint, below the wall tops."""
        c = self.cube_pos()
        d = np.abs(c[:2] - self.basket_pos[:2])
        return bool(d[0] < BASKET_HALF and d[1] < BASKET_HALF and c[2] < 0.05)

    # ── rendering ───────────────────────────────────────────────────────────
    def render(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.render_size)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
