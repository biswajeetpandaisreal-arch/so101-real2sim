"""Kinematic check of the digital twin against a tape measure.

Teleoperate the follower so the tip of its fixed jaw touches the centre of each
taped cross, and press Enter. The follower's joint angles are logged and pushed
through the MuJoCo model's forward kinematics; the predicted tip position is
compared with the tape-measured one. Two errors are reported:

  raw      model tip vs tape measurement, as-is
  aligned  after the best 2-D rigid fit (rotation + translation), which removes
           error in where the tape frame was laid down and leaves the error of
           the kinematic chain itself (link lengths, joint zeros, calibration)

    python scripts/touch_spots.py              # Enter = record, s = skip, q = quit
    python scripts/touch_spots.py --analyse data/touch/touch_<time>.json
"""

import argparse
import json
import msvcrt
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.layout import read_layout  # noqa: E402
from real2sim.sim_env import FPS, JOINTS, SO101Env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "config" / "calibration"
TIP_GEOM = "fixed_jaw_sph_tip1"              # very tip of the fixed jaw in so101.xml


def tip_position(env, q_deg):
    env.data.qpos[env.qadr] = env.lerobot_to_rad(np.asarray(q_deg, float))
    mujoco.mj_kinematics(env.model, env.data)
    g = env.model.geom(TIP_GEOM)
    return env.data.geom_xpos[g.id] - np.array([0, 0, g.size[0]])   # bottom of the tip sphere


def rigid_fit_2d(P, Q):
    """Rotation R and translation t minimising |R P + t - Q| (Kabsch, 2-D)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def analyse(rec):
    env = SO101Env()
    pts = [r for r in rec["points"]]
    meas = np.array([[*r["xy_measured"], 0.0] for r in pts])
    model = np.array([tip_position(env, r["q_deg"]) for r in pts])
    err = (model - meas) * 1000
    R, t = rigid_fit_2d(model[:, :2], meas[:, :2])
    aligned = (model[:, :2] @ R.T + t - meas[:, :2]) * 1000
    print(f"{'spot':6s} {'dx':>6s} {'dy':>6s} {'dz':>6s} {'|xy|':>6s} {'aligned':>8s}   (mm, model - tape)")
    for r, e, a in zip(pts, err, aligned):
        print(f"{r['name']:6s} {e[0]:6.1f} {e[1]:6.1f} {e[2]:6.1f} {np.hypot(*e[:2]):6.1f} {np.hypot(*a):8.1f}")
    rms = lambda v: float(np.sqrt(np.mean(np.sum(v ** 2, axis=1))))   # noqa: E731
    summary = {
        "n_points": len(pts),
        "raw_xy_rms_mm": rms(err[:, :2]), "raw_xy_max_mm": float(np.hypot(err[:, 0], err[:, 1]).max()),
        "mean_bias_mm": err.mean(0).round(2).tolist(),
        "z_rms_mm": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "aligned_xy_rms_mm": rms(aligned),
        "fit_rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        "fit_translation_mm": (t * 1000).round(2).tolist(),
    }
    print(json.dumps(summary, indent=1))
    return summary


def record(out):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    layout = {k: v for k, v in read_layout().items() if v["role"] != "basket"}
    leader = SO101Leader(SO101LeaderConfig(port="COM4", id="so101_leader", calibration_dir=CALIB))
    robot = SO101Follower(SO101FollowerConfig(port="COM3", id="so101_follower",
                                              calibration_dir=CALIB, max_relative_target=10))
    leader.connect()
    robot.connect()
    points = []
    try:
        for name, spot in layout.items():
            print(f"\nTouch the fixed-jaw tip to the centre of cross {name} "
                  f"({spot['role']}), then press Enter.  s = skip, q = quit")
            key = None
            while key not in ("\r", "s", "q"):
                t0 = time.perf_counter()
                robot.send_action(leader.get_action())
                if msvcrt.kbhit():
                    key = msvcrt.getwch()
                time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
            if key == "q":
                break
            if key == "s":
                continue
            reads = [robot.bus.sync_read("Present_Position") for _ in range(10)]
            raw = robot.bus.sync_read("Present_Position", normalize=False)
            q = np.mean([[r[j] for j in JOINTS] for r in reads], axis=0)
            points.append({"name": name, "role": spot["role"], "xy_measured": spot["xy"].tolist(),
                           "q_deg": q.tolist(), "raw_counts": [int(raw[j]) for j in JOINTS],
                           "time": time.time()})
            print(f"  recorded {name}")
    finally:
        print("\nMoving on: lift the leader so the follower clears the table, then let it rest.")
        robot.disconnect()
        leader.disconnect()
    rec = {"tip_geom": TIP_GEOM, "points": points}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(f"saved {out}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", default=None, help="re-analyse a saved recording")
    a = ap.parse_args()
    if a.analyse:
        rec = json.loads(Path(a.analyse).read_text())
    else:
        rec = record(ROOT / "data" / "touch" / f"touch_{time.strftime('%Y%m%d_%H%M%S')}.json")
    if len(rec["points"]) >= 3:
        analyse(rec)


if __name__ == "__main__":
    main()
