import importlib.util
import json
from pathlib import Path
import sys
import threading
import types
from unittest import mock

import numpy as np
import pytest

from sensor.action_filter import ActionFilterResult, ActionProbabilityFilter


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeDuration:
    def __init__(self, seconds):
        self.seconds = float(seconds)

    def to_sec(self):
        return self.seconds


class FakeTime:
    now_sec = 0.0

    def __init__(self, seconds=0.0):
        self.seconds = float(seconds)

    @classmethod
    def now(cls):
        return cls(cls.now_sec)

    def to_sec(self):
        return self.seconds

    def to_nsec(self):
        return int(self.seconds * 1_000_000_000)

    def __sub__(self, other):
        return FakeDuration(self.seconds - other.seconds)

    def __eq__(self, other):
        return isinstance(other, FakeTime) and self.seconds == other.seconds


class FakeInt32:
    def __init__(self):
        self.data = 0


class FakeUInt8:
    def __init__(self, data=0):
        self.data = int(data)


def load_bridge_module():
    rospy = types.ModuleType("rospy")
    rospy.Time = FakeTime
    rospy.loginfo = lambda *args: None
    rospy.loginfo_throttle = lambda *args: None
    rospy.logwarn_throttle = lambda *args: None
    rospy.logwarn = lambda *args: None
    rospy.logerr_throttle = lambda *args: None
    rospy.signal_shutdown = lambda *args: None

    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})

    gym = types.ModuleType("gym")
    gym_spaces = types.ModuleType("gym.spaces")
    gym_spaces.Box = type("Box", (), {})
    gym_spaces.Dict = type("Dict", (), {})
    gym_spaces.Discrete = type("Discrete", (), {})
    gym.spaces = gym_spaces

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PointStamped = type("PointStamped", (), {})
    geometry_msgs.msg = geometry_msgs_msg

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = type("Image", (), {})
    sensor_msgs.msg = sensor_msgs_msg

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Header = type("Header", (), {})
    std_msgs_msg.Int32 = FakeInt32
    std_msgs_msg.UInt8 = FakeUInt8
    std_msgs.msg = std_msgs_msg

    habitat_baselines = types.ModuleType("habitat_baselines")
    habitat_rl = types.ModuleType("habitat_baselines.rl")
    habitat_ddppo = types.ModuleType("habitat_baselines.rl.ddppo")
    habitat_policy = types.ModuleType("habitat_baselines.rl.ddppo.policy")
    habitat_policy.PointNavResNetPolicy = type("PointNavResNetPolicy", (), {})
    habitat_utils = types.ModuleType("habitat_baselines.utils")
    habitat_common = types.ModuleType("habitat_baselines.utils.common")
    habitat_common.batch_obs = lambda *args, **kwargs: None

    module_path = REPO_ROOT / "sensor" / "falcon_ros_bridge.py"
    spec = importlib.util.spec_from_file_location(
        "falcon_ros_bridge_for_safety_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    modules = {
        "rospy": rospy,
        "torch": torch,
        "gym": gym,
        "gym.spaces": gym_spaces,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "habitat_baselines": habitat_baselines,
        "habitat_baselines.rl": habitat_rl,
        "habitat_baselines.rl.ddppo": habitat_ddppo,
        "habitat_baselines.rl.ddppo.policy": habitat_policy,
        "habitat_baselines.utils": habitat_utils,
        "habitat_baselines.utils.common": habitat_common,
    }
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


BRIDGE_MODULE = load_bridge_module()


class FillValue:
    def __init__(self, value):
        self.value = value

    def fill_(self, value):
        self.value = int(value)
        return self


class RecordingPublisher:
    def __init__(self, events=None):
        self.values = []
        self.events = events

    def publish(self, msg):
        self.values.append(int(msg.data))
        if self.events is not None:
            self.events.append(("publish", int(msg.data)))


def make_filter():
    action_filter = ActionProbabilityFilter(
        action_count=4,
        tau_sec=0.0,
        switch_margin=0.0,
        switch_hold_sec=0.0,
        stop_hold_sec=0.0,
    )
    action_filter.force_action(0)
    return action_filter


def make_bridge_node(events=None):
    node = BRIDGE_MODULE.FalconRosBridge.__new__(BRIDGE_MODULE.FalconRosBridge)
    node.action_state_lock = threading.RLock()
    node.action_filter_enabled = True
    node.action_filter = make_filter()
    node.safety_stop_generation = 0
    node.pending_prev_stop = False
    node.tag_search_enabled = False
    node.tracking_state_timeout_sec = 0.5
    node.tag_search_timeout_sec = 12.0
    node.tag_search_default_action = BRIDGE_MODULE.SEARCH_LEFT_ACTION_ID
    node.tracking_state = 0
    node.tracking_state_receive_time = None
    node.have_seen_visible_tracking_state = False
    node.search_active = False
    node.search_started_at = None
    node.search_action_id = BRIDGE_MODULE.SEARCH_LEFT_ACTION_ID
    node.search_rearm_required = False
    node.prev_actions = FillValue(3)
    node.last_obs_time = FakeTime(1.0)
    node.stopped_for_data_timeout = False
    node.data_timeout_sec = 0.3
    node.action_pub = RecordingPublisher(events)
    node.debug_mapping = False
    node.debug_depth = False
    node.debug_timing = False
    node.replay_dump_enabled = False
    node.action_topic = "/falcon/action_id"
    node.latest_polar_msg = None
    node.depth_key = "depth"
    node.goal_key = "goal"
    node._build_obs = lambda **kwargs: (
        {
            "depth": np.zeros((2, 2, 1), dtype=np.float32),
            "goal": np.array([1.0, 0.0], dtype=np.float32),
        },
        {},
    )
    node._emit_heartbeat = lambda: None
    return node


def make_messages():
    depth_msg = types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=FakeTime(1.0))
    )
    polar_msg = types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=FakeTime(1.0)),
        point=types.SimpleNamespace(x=1.0, y=0.0),
    )
    return depth_msg, polar_msg


def test_watchdog_stop_during_inference_discards_stale_motion_and_replay():
    node = make_bridge_node()
    depth_msg, polar_msg = make_messages()
    inference_started = threading.Event()
    allow_inference_to_finish = threading.Event()
    replay_calls = []

    def infer(_obs):
        inference_started.set()
        assert allow_inference_to_finish.wait(timeout=2.0)
        return (
            1,
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            {},
        )

    node._infer_action = infer
    node._maybe_dump_replay_sample = lambda **kwargs: replay_calls.append(
        kwargs
    )

    worker = threading.Thread(
        target=node._process_one, args=(depth_msg, polar_msg)
    )
    worker.start()
    assert inference_started.wait(timeout=2.0)

    FakeTime.now_sec = 1.31
    node._watchdog_cb(None)
    allow_inference_to_finish.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert node.action_pub.values == [0]
    assert replay_calls == []
    assert node.prev_actions.value == 0
    assert node.action_filter.stable_action == 0
    assert node.safety_stop_generation == 1
    assert node.stopped_for_data_timeout


def test_process_publishes_and_commits_state_before_replay_dump():
    events = []
    node = make_bridge_node(events)
    depth_msg, polar_msg = make_messages()
    node.action_filter_enabled = False
    node._infer_action = lambda _obs: (
        3,
        None,
        {
            "hidden_in": np.zeros((1, 1), dtype=np.float32),
            "prev_action_in": np.array([[0]], dtype=np.int64),
            "not_done_mask_in": np.array([[False]]),
        },
    )
    node._emit_heartbeat = lambda: events.append(("heartbeat", None))

    def dump(**kwargs):
        events.append(
            (
                "dump",
                kwargs["raw_action_id"],
                kwargs["action_id"],
                node.prev_actions.value,
                node.last_obs_time.seconds,
            )
        )
        return False

    node._maybe_dump_replay_sample = dump
    FakeTime.now_sec = 2.0

    node._process_one(depth_msg, polar_msg)

    assert events == [
        ("publish", 3),
        ("heartbeat", None),
        ("dump", 3, 3, 3, 2.0),
    ]
    assert not node.stopped_for_data_timeout


def test_search_is_opt_in_and_holds_stop_when_disabled():
    node = make_bridge_node()
    depth_msg, _ = make_messages()

    with mock.patch.object(BRIDGE_MODULE.time, "monotonic", return_value=1.0):
        node._tracking_state_cb(FakeUInt8(BRIDGE_MODULE.TRACKING_STATE_SEARCHABLE))
        node._process_one(depth_msg)

    assert node.action_pub.values == [0]
    assert not node.search_active
    assert node.prev_actions.value == 3


def test_search_entry_stops_then_turns_without_entering_policy_history():
    node = make_bridge_node()
    node.tag_search_enabled = True
    depth_msg, polar_msg = make_messages()
    polar_msg.point.y = -0.2
    node.latest_polar_msg = polar_msg

    with mock.patch.object(
        BRIDGE_MODULE.time, "monotonic", side_effect=[0.9, 1.0, 1.1]
    ):
        node._tracking_state_cb(FakeUInt8(BRIDGE_MODULE.TRACKING_STATE_VISIBLE))
        node._tracking_state_cb(FakeUInt8(BRIDGE_MODULE.TRACKING_STATE_SEARCHABLE))
        node._process_one(depth_msg)

    assert node.action_pub.values == [0, BRIDGE_MODULE.SEARCH_RIGHT_ACTION_ID]
    assert node.search_active
    assert node.prev_actions.value == 3
    assert node.pending_prev_stop
    assert node.action_filter.stable_action == 0


def test_search_exit_stops_then_next_depth_resumes_policy_from_reset_state():
    node = make_bridge_node()
    node.tag_search_enabled = True
    depth_msg, polar_msg = make_messages()
    node.latest_polar_msg = polar_msg
    node.action_filter_enabled = False
    node._infer_action = lambda _obs: (1, None, {})
    node._maybe_dump_replay_sample = lambda **kwargs: False

    with mock.patch.object(
        BRIDGE_MODULE.time,
        "monotonic",
        side_effect=[0.9, 1.0, 1.1, 1.2, 1.3],
    ):
        node._tracking_state_cb(FakeUInt8(BRIDGE_MODULE.TRACKING_STATE_VISIBLE))
        node._tracking_state_cb(FakeUInt8(BRIDGE_MODULE.TRACKING_STATE_SEARCHABLE))
        node._process_one(depth_msg)
        node._tracking_state_cb(FakeUInt8(BRIDGE_MODULE.TRACKING_STATE_VISIBLE))
        node._process_one(depth_msg, polar_msg)

    assert node.action_pub.values == [
        0,
        BRIDGE_MODULE.SEARCH_LEFT_ACTION_ID,
        0,
        1,
    ]
    assert not node.search_active
    assert not node.pending_prev_stop
    assert node.prev_actions.value == 1


def test_expired_search_cannot_restart_on_same_state_heartbeat():
    node = make_bridge_node()
    node.tag_search_enabled = True
    depth_msg, _ = make_messages()

    with mock.patch.object(
        BRIDGE_MODULE.time,
        "monotonic",
        side_effect=[0.0, 0.1, 0.2, 12.2, 12.3, 12.4],
    ):
        node._tracking_state_cb(FakeUInt8(BRIDGE_MODULE.TRACKING_STATE_VISIBLE))
        node._tracking_state_cb(FakeUInt8(BRIDGE_MODULE.TRACKING_STATE_SEARCHABLE))
        node._process_one(depth_msg)
        node._process_one(depth_msg)
        node._tracking_state_cb(FakeUInt8(BRIDGE_MODULE.TRACKING_STATE_SEARCHABLE))
        node._process_one(depth_msg)

    assert node.action_pub.values == [
        0,
        BRIDGE_MODULE.SEARCH_LEFT_ACTION_ID,
        0,
    ]
    assert not node.search_active
    assert node.search_rearm_required


def test_enabled_search_mode_rejects_stale_tracking_state():
    node = make_bridge_node()
    node.tag_search_enabled = True
    node.tracking_state = BRIDGE_MODULE.TRACKING_STATE_VISIBLE
    node.tracking_state_receive_time = 1.0
    depth_msg, polar_msg = make_messages()
    node._infer_action = lambda _obs: pytest.fail("stale state reached policy")

    with mock.patch.object(BRIDGE_MODULE.time, "monotonic", return_value=1.6):
        node._process_one(depth_msg, polar_msg)

    assert node.action_pub.values == [0]


def test_replay_dump_distinguishes_raw_and_executed_actions(tmp_path):
    node = make_bridge_node()
    depth_msg, polar_msg = make_messages()
    node.replay_dump_enabled = True
    node.replay_dump_dir = str(tmp_path)
    node.replay_dump_limit = 5
    node._replay_dump_count = 0
    node._replay_dump_limit_reached = False
    node.resolution = 2
    node.max_depth_m = 10.0
    node.deterministic = True
    obs = {
        "depth": np.full((2, 2, 1), 0.25, dtype=np.float32),
        "goal": np.array([2.0, -0.5], dtype=np.float32),
    }
    recurrent_input = {
        "hidden_in": np.zeros((1, 2), dtype=np.float32),
        "prev_action_in": np.array([[2]], dtype=np.int64),
        "not_done_mask_in": np.array([[True]]),
    }
    probs = np.array([0.05, 0.80, 0.10, 0.05], dtype=np.float32)
    filter_result = ActionFilterResult(
        action_id=3,
        raw_argmax=1,
        candidate_action=3,
        switched=False,
        alpha=0.2,
        filtered_probs=(0.05, 0.20, 0.10, 0.65),
    )
    FakeTime.now_sec = 12.5

    reached_limit = node._maybe_dump_replay_sample(
        obs=obs,
        raw_action_id=1,
        action_id=3,
        probs=probs,
        filter_result=filter_result,
        recurrent_input=recurrent_input,
        depth_debug={"encoding": "16UC1"},
        depth_msg=depth_msg,
        polar_msg=polar_msg,
    )

    assert not reached_limit
    npz_path = next(tmp_path.glob("*.npz"))
    json_path = next(tmp_path.glob("*.json"))
    with np.load(npz_path) as replay:
        assert replay["action"].tolist() == [3]
        assert replay["raw_action"].tolist() == [1]
        assert replay["prev_action_in"].tolist() == [[2]]
        assert replay["probs"] == pytest.approx(probs)
        assert replay["filtered_probs"] == pytest.approx(
            filter_result.filtered_probs
        )
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["action_id"] == 3
    assert meta["action_name"] == "right"
    assert meta["raw_action_id"] == 1
    assert meta["raw_action_name"] == "forward"
    assert meta["filtered_action_probs"] == pytest.approx(
        filter_result.filtered_probs
    )
    assert meta["action_filter_switched"] is False
