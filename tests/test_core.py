"""Fast checks of the pieces everything else depends on (no datasets, no hardware).

    python -m pytest tests
"""

import numpy as np
import pytest

from real2sim.demos import active_span, grasp_frame
from real2sim.generate import displacement_weight, segment, wrap90
from real2sim.kinematics import SO101IK, min_jerk
from real2sim.layout import read_layout, to_sim, to_table_cm
from real2sim.sim_env import FPS, SO101Env
from real2sim.stats import wilson


@pytest.fixture(scope="module")
def env():
    return SO101Env()


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(3, 3)
    assert hi == 100.0 and 40 < lo < 50                 # 3/3 is compatible with ~44-100%
    lo, hi = wilson(50, 100)
    assert lo < 50 < hi and hi - lo < 20


def test_table_frame_round_trip():
    assert np.allclose(to_table_cm(to_sim(30.0, -5.0)), [30.0, -5.0])


def test_units_round_trip(env):
    a = np.array([10.0, -20.0, 30.0, -40.0, 50.0, 60.0])
    assert np.allclose(env.rad_to_lerobot(env.lerobot_to_rad(a)), a)


def test_measured_joint_limits_cover_rest_pose(env):
    # the real follower rests at wrist_flex -101.4 deg, outside the Menagerie limit of 95 deg
    assert np.rad2deg(env.range[3, 1]) > 101


def test_object_settles_and_success_needs_the_box(env):
    env.reset(read_layout()["1"]["xy"])
    assert abs(env.object_pos()[2] - env.obj_half_height) < 2e-3 and not env.success()
    env.reset(env.box_pos[:2])
    for _ in range(30):
        env.step(env.state_lerobot())
    assert env.success()


def test_ik_reaches_every_mark(env):
    ik = SO101IK(env)
    for mark in read_layout().values():
        if mark["role"] != "box":
            _, err = ik.solve(np.r_[mark["xy"], 0.012])
            assert err < 0.005


def test_segment_skips_the_closed_gripper_at_rest():
    g = np.r_[np.full(30, 1.0), np.full(60, 60.0), np.full(60, 1.0), np.full(30, 60.0), np.full(30, 1.0)]
    assert segment(np.c_[np.zeros((len(g), 5)), g]) == (90, 150)


def test_displacement_is_complete_before_the_descent_and_gone_after_release():
    w = displacement_weight(300, t_grasp=150, t_release=250, t_descent=120)
    assert w[0] == 0 and w[120] == pytest.approx(1) and w[150] == pytest.approx(1) and w[-1] == 0


def test_grasp_frame_and_active_span():
    T = 200
    action, state = np.zeros((T, 6)), np.zeros((T, 6))
    state[60:, 1] = 30.0                                  # the arm sets off at frame 60
    action[:, 5], state[:, 5] = 50.0, 50.0
    action[100:, 5], state[100:, 5] = 1.0, 15.0           # jaws stall on the object from frame 100
    assert grasp_frame(action, state) == 100
    assert active_span(state, pad=0)[0] == 60


def test_helpers():
    assert min_jerk(0.0, 2.0, 0.5) == pytest.approx(1.0)
    assert wrap90(np.pi / 2) == pytest.approx(0.0)
    assert FPS == 30
