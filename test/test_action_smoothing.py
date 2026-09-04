import importlib.util
import math
from pathlib import Path
import sys
import types
import threading
from unittest import mock

import pytest

from sensor.action_filter import ActionProbabilityFilter


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_unitree_mapper_module():
    rospy = types.ModuleType("rospy")
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Int32 = type("Int32", (), {})
    std_msgs.msg = std_msgs_msg

    module_path = REPO_ROOT / "go2" / "unitree_action_mapper.py"
    spec = importlib.util.spec_from_file_location(
        "unitree_mapper_for_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"rospy": rospy, "std_msgs": std_msgs, "std_msgs.msg": std_msgs_msg},
    ):
        spec.loader.exec_module(module)
    return module


MAPPER_MODULE = load_unitree_mapper_module()
VelocitySlewLimiter = MAPPER_MODULE.VelocitySlewLimiter


def test_deployed_mapper_copy_matches_direct_launcher_copy():
    direct = (REPO_ROOT / "go2" / "unitree_action_mapper.py").read_bytes()
    catkin = (
        REPO_ROOT
        / "ranger_ws"
        / "src"
        / "go_nav"
        / "scripts"
        / "unitree_action_mapper.py"
    ).read_bytes()
    assert direct == catkin


def make_action_filter(tau_sec=0.15):
    action_filter = ActionProbabilityFilter(
        action_count=4,
        tau_sec=tau_sec,
        switch_margin=0.10,
        switch_hold_sec=0.12,
        stop_hold_sec=0.20,
    )
    action_filter.force_action(3)
    return action_filter


def test_action_ema_rejects_a_single_frame_spike():
    action_filter = make_action_filter()

    assert action_filter.update([0.0, 0.05, 0.05, 0.90], 0.00).action_id == 3
    assert action_filter.update([0.0, 0.90, 0.05, 0.05], 0.05).action_id == 3
    result = action_filter.update([0.0, 0.05, 0.05, 0.90], 0.10)

    assert result.action_id == 3
    assert result.candidate_action == 3


def test_sustained_probability_change_eventually_switches_with_time_based_ema():
    action_filter = make_action_filter()
    action_filter.update([0.0, 0.05, 0.05, 0.90], 0.00)

    results = [
        action_filter.update([0.0, 0.90, 0.05, 0.05], timestamp)
        for timestamp in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
    ]

    assert results[0].alpha == pytest.approx(1.0 - math.exp(-0.05 / 0.15))
    assert all(result.action_id == 3 for result in results[:5])
    assert results[-1].action_id == 1
    assert sum(result.switched for result in results) == 1


def test_hysteresis_requires_sustained_motion_and_longer_stop_confirmation():
    action_filter = make_action_filter(tau_sec=0.0)

    assert action_filter.update([0.0, 0.90, 0.05, 0.05], 0.00).action_id == 3
    assert action_filter.update([0.0, 0.90, 0.05, 0.05], 0.119).action_id == 3
    assert action_filter.update([0.0, 0.90, 0.05, 0.05], 0.121).action_id == 1

    assert action_filter.update([0.95, 0.02, 0.02, 0.01], 0.20).action_id == 1
    assert action_filter.update([0.95, 0.02, 0.02, 0.01], 0.399).action_id == 1
    assert action_filter.update([0.95, 0.02, 0.02, 0.01], 0.401).action_id == 0


def test_force_stop_clears_filter_history_and_requires_reconfirmation():
    action_filter = make_action_filter(tau_sec=0.0)
    action_filter.update([0.0, 0.90, 0.05, 0.05], 0.00)

    action_filter.force_action(0)
    first_recovery = action_filter.update([0.0, 0.90, 0.05, 0.05], 1.00)
    confirmed_recovery = action_filter.update([0.0, 0.90, 0.05, 0.05], 1.121)

    assert first_recovery.action_id == 0
    assert confirmed_recovery.action_id == 1


@pytest.mark.parametrize(
    "probabilities",
    (
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5],
        [0.5, -0.1, 0.3, 0.3],
        [0.5, math.nan, 0.2, 0.3],
    ),
)
def test_action_filter_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        make_action_filter().update(probabilities, 0.0)


def make_velocity_limiter():
    return VelocitySlewLimiter(
        linear_accel_limit=1.0,
        linear_decel_limit=1.5,
        yaw_accel_limit=2.0,
        yaw_decel_limit=3.0,
    )


class FakeDuration:
    def __init__(self, seconds):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds


class FakeTime:
    now_sec = 0.0

    def __init__(self, seconds=0.0):
        self.seconds = float(seconds)

    @classmethod
    def now(cls):
        return cls(cls.now_sec)

    def __sub__(self, other):
        return FakeDuration(self.seconds - other.seconds)

    def __eq__(self, other):
        return isinstance(other, FakeTime) and self.seconds == other.seconds


class FakeController:
    def __init__(self):
        self.moves = []
        self.stop_count = 0

    def move(self, vx, vy, vyaw):
        self.moves.append((vx, vy, vyaw))

    def stop(self):
        self.stop_count += 1


def make_mapper_node():
    node = MAPPER_MODULE.UnitreeActionMapper.__new__(
        MAPPER_MODULE.UnitreeActionMapper
    )
    node.args = types.SimpleNamespace(
        action_timeout_sec=0.3, watchdog_rate_hz=20.0
    )
    node.controller = FakeController()
    node.action_to_velocity = {
        0: (0.0, 0.0, 0.0),
        1: (0.6, 0.0, 0.0),
        2: (0.0, 0.0, 0.6),
        3: (0.0, 0.0, -0.6),
        4: (0.0, 0.0, 0.25),
        5: (0.0, 0.0, -0.25),
    }
    node.last_action_time = FakeTime(0)
    node.last_action_id = None
    node.stopped_for_timeout = False
    node.state_lock = threading.RLock()
    node.sdk_call_lock = threading.Lock()
    node.command_generation = 0
    node.pending_stop_calls = 0
    node.shutdown_requested = False
    node.smoothing_enabled = True
    node.velocity_limiter = make_velocity_limiter()
    node.last_control_time = FakeTime(0)
    node.args.stop_on_shutdown = True
    return node


def test_velocity_slew_limits_each_control_tick():
    limiter = make_velocity_limiter()
    limiter.set_target(0.6, 0.0, -0.6)

    vx, vy, vyaw = limiter.step(0.04)

    assert vx == pytest.approx(0.04)
    assert vy == 0.0
    assert vyaw == pytest.approx(-0.08)


def test_velocity_reversal_brakes_through_zero():
    limiter = make_velocity_limiter()
    limiter.current = (0.0, 0.0, -0.20)
    limiter.set_target(0.0, 0.0, 0.60)

    assert limiter.step(0.04)[2] == pytest.approx(-0.08)
    assert limiter.step(0.04)[2] == pytest.approx(0.0)
    assert limiter.step(0.04)[2] == pytest.approx(0.08)


def test_velocity_limiter_lands_on_target_and_reset_is_immediate():
    limiter = make_velocity_limiter()
    limiter.set_target(0.12, 0.0, 0.10)

    for _ in range(10):
        velocity = limiter.step(0.05)

    assert velocity == pytest.approx((0.12, 0.0, 0.10))
    limiter.reset()
    assert limiter.current == (0.0, 0.0, 0.0)
    assert limiter.target == (0.0, 0.0, 0.0)


def test_mapper_ramps_motion_but_action_zero_stops_immediately(monkeypatch):
    monkeypatch.setattr(MAPPER_MODULE.rospy, "Time", FakeTime, raising=False)
    monkeypatch.setattr(
        MAPPER_MODULE.rospy,
        "loginfo_throttle",
        lambda *args: None,
        raising=False,
    )
    monkeypatch.setattr(
        MAPPER_MODULE.rospy,
        "logwarn_throttle",
        lambda *args: None,
        raising=False,
    )
    monkeypatch.setattr(
        MAPPER_MODULE.rospy, "logdebug", lambda *args: None, raising=False
    )
    node = make_mapper_node()

    FakeTime.now_sec = 0.01
    node.action_cb(types.SimpleNamespace(data=1))
    assert node.controller.moves == []

    FakeTime.now_sec = 0.05
    node.control_cb(None)
    assert node.controller.moves[-1] == pytest.approx((0.05, 0.0, 0.0))

    FakeTime.now_sec = 0.06
    node.action_cb(types.SimpleNamespace(data=0))
    assert node.controller.stop_count == 1
    assert node.velocity_limiter.current == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("action_id", "expected_vyaw"), ((4, 0.25), (5, -0.25))
)
def test_mapper_uses_separate_low_speed_search_actions(
    monkeypatch, action_id, expected_vyaw
):
    configure_mapper_logging_stubs(monkeypatch)
    node = make_mapper_node()

    FakeTime.now_sec = 0.01
    node.action_cb(types.SimpleNamespace(data=action_id))
    assert node.velocity_limiter.target == (0.0, 0.0, expected_vyaw)

    FakeTime.now_sec = 0.05
    node.control_cb(None)
    assert node.controller.moves[-1][0:2] == (0.0, 0.0)
    assert math.copysign(1.0, node.controller.moves[-1][2]) == math.copysign(
        1.0, expected_vyaw
    )
    assert abs(node.controller.moves[-1][2]) <= abs(expected_vyaw)


def test_mapper_watchdog_stops_once_and_resets_limiter(monkeypatch):
    monkeypatch.setattr(MAPPER_MODULE.rospy, "Time", FakeTime, raising=False)
    monkeypatch.setattr(
        MAPPER_MODULE.rospy,
        "loginfo_throttle",
        lambda *args: None,
        raising=False,
    )
    monkeypatch.setattr(
        MAPPER_MODULE.rospy,
        "logwarn_throttle",
        lambda *args: None,
        raising=False,
    )
    monkeypatch.setattr(
        MAPPER_MODULE.rospy, "logwarn", lambda *args: None, raising=False
    )
    monkeypatch.setattr(
        MAPPER_MODULE.rospy, "logdebug", lambda *args: None, raising=False
    )
    node = make_mapper_node()

    FakeTime.now_sec = 1.00
    node.action_cb(types.SimpleNamespace(data=3))
    FakeTime.now_sec = 1.05
    node.control_cb(None)
    assert node.controller.moves

    FakeTime.now_sec = 1.31
    node.control_cb(None)
    FakeTime.now_sec = 1.40
    node.control_cb(None)

    assert node.controller.stop_count == 1
    assert node.stopped_for_timeout
    assert node.velocity_limiter.current == (0.0, 0.0, 0.0)


def configure_mapper_logging_stubs(monkeypatch):
    for name in (
        "logdebug",
        "logerr",
        "loginfo_throttle",
        "logwarn",
        "logwarn_throttle",
    ):
        monkeypatch.setattr(
            MAPPER_MODULE.rospy, name, lambda *args: None, raising=False
        )
    monkeypatch.setattr(MAPPER_MODULE.rospy, "Time", FakeTime, raising=False)


class StateLockCheckingController(FakeController):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.state_lock_was_free = []

    def _check_state_lock(self):
        acquired = self.node.state_lock.acquire(blocking=False)
        self.state_lock_was_free.append(acquired)
        if acquired:
            self.node.state_lock.release()

    def move(self, vx, vy, vyaw):
        self._check_state_lock()
        super().move(vx, vy, vyaw)

    def stop(self):
        self._check_state_lock()
        super().stop()


def test_mapper_never_holds_state_lock_during_sdk_calls(monkeypatch):
    configure_mapper_logging_stubs(monkeypatch)
    node = make_mapper_node()
    # A non-reentrant lock lets the fake SDK detect same-thread lock ownership.
    node.state_lock = threading.Lock()
    node.controller = StateLockCheckingController(node)

    FakeTime.now_sec = 0.01
    node.action_cb(types.SimpleNamespace(data=1))
    FakeTime.now_sec = 0.05
    node.control_cb(None)

    FakeTime.now_sec = 0.06
    node.action_cb(types.SimpleNamespace(data=0))
    FakeTime.now_sec = 0.07
    node.action_cb(types.SimpleNamespace(data=99))

    FakeTime.now_sec = 1.00
    node.action_cb(types.SimpleNamespace(data=1))
    FakeTime.now_sec = 1.31
    node.control_cb(None)
    node.shutdown()

    assert node.controller.moves
    assert node.controller.stop_count == 4
    assert node.controller.state_lock_was_free
    assert all(node.controller.state_lock_was_free)


class BlockingMoveController(FakeController):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.first_move_entered = threading.Event()
        self.release_first_move = threading.Event()

    def move(self, vx, vy, vyaw):
        velocity = (vx, vy, vyaw)
        self.calls.append(("move_start", velocity))
        if not self.moves:
            self.first_move_entered.set()
            if not self.release_first_move.wait(timeout=2.0):
                raise RuntimeError(
                    "test timed out waiting to release first Move"
                )
        self.moves.append(velocity)
        self.calls.append(("move_end", velocity))

    def stop(self):
        self.stop_count += 1
        self.calls.append(("stop", None))


def test_stop_invalidates_a_move_already_waiting_for_the_sdk(monkeypatch):
    configure_mapper_logging_stubs(monkeypatch)
    node = make_mapper_node()
    node.controller = BlockingMoveController()

    move_attempt_count = 0
    move_attempt_lock = threading.Lock()
    second_move_prepared = threading.Event()
    original_execute_move = node._execute_move

    def marked_execute_move(move_generation, velocity):
        nonlocal move_attempt_count
        with move_attempt_lock:
            move_attempt_count += 1
            if move_attempt_count == 2:
                second_move_prepared.set()
        return original_execute_move(move_generation, velocity)

    stop_prepared = threading.Event()
    original_execute_stop = node._execute_stop

    def marked_execute_stop():
        stop_prepared.set()
        return original_execute_stop()

    node._execute_move = marked_execute_move
    node._execute_stop = marked_execute_stop

    FakeTime.now_sec = 0.01
    node.action_cb(types.SimpleNamespace(data=1))
    FakeTime.now_sec = 0.05
    first_move = threading.Thread(target=node.control_cb, args=(None,))
    first_move.start()
    assert node.controller.first_move_entered.wait(timeout=1.0)

    FakeTime.now_sec = 0.10
    queued_move = threading.Thread(target=node.control_cb, args=(None,))
    queued_move.start()
    assert second_move_prepared.wait(timeout=1.0)

    FakeTime.now_sec = 0.11
    safety_stop = threading.Thread(
        target=node.action_cb, args=(types.SimpleNamespace(data=0),)
    )
    safety_stop.start()
    assert stop_prepared.wait(timeout=1.0)
    assert node.command_generation == 2
    assert node.pending_stop_calls == 1

    node.controller.release_first_move.set()
    for thread in (first_move, queued_move, safety_stop):
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    # Only the in-flight Move may finish. The queued stale Move is discarded,
    # and StopMove is the final SDK command.
    assert len(node.controller.moves) == 1
    assert node.controller.stop_count == 1
    assert node.controller.calls[-1] == ("stop", None)
    assert node.pending_stop_calls == 0
