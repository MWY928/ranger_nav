import importlib.util
import math
import sys
import threading
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def install_ros_import_stubs():
    rospy = types.ModuleType("rospy")

    class FakeDuration:
        def __init__(self, secs=0.0):
            self.secs = float(secs)

        @classmethod
        def from_sec(cls, secs):
            return cls(secs)

        def to_sec(self):
            return self.secs

    class FakeTime:
        current_sec = 0.0

        def __init__(self, secs=0.0):
            self.secs = float(secs)

        @classmethod
        def now(cls):
            return cls(cls.current_sec)

        def __eq__(self, other):
            return isinstance(other, FakeTime) and self.secs == other.secs

        def __sub__(self, other):
            return FakeDuration(self.secs - other.secs)

        def __add__(self, duration):
            return FakeTime(self.secs + duration.to_sec())

        def to_sec(self):
            return self.secs

    rospy.Time = FakeTime
    rospy.Duration = FakeDuration
    rospy.loginfo_throttle = lambda *args: None
    rospy.logwarn_throttle = lambda *args: None
    sys.modules.setdefault("rospy", rospy)

    apriltag_msg = types.ModuleType("apriltag_ros.msg")
    apriltag_msg.AprilTagDetectionArray = type("AprilTagDetectionArray", (), {})
    apriltag_pkg = types.ModuleType("apriltag_ros")
    apriltag_pkg.msg = apriltag_msg
    sys.modules.setdefault("apriltag_ros", apriltag_pkg)
    sys.modules.setdefault("apriltag_ros.msg", apriltag_msg)

    geometry_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msg.PointStamped = type("PointStamped", (), {})
    geometry_pkg = types.ModuleType("geometry_msgs")
    geometry_pkg.msg = geometry_msg
    sys.modules.setdefault("geometry_msgs", geometry_pkg)
    sys.modules.setdefault("geometry_msgs.msg", geometry_msg)

    nav_msg = types.ModuleType("nav_msgs.msg")
    nav_msg.Odometry = type("Odometry", (), {})
    nav_pkg = types.ModuleType("nav_msgs")
    nav_pkg.msg = nav_msg
    sys.modules.setdefault("nav_msgs", nav_pkg)
    sys.modules.setdefault("nav_msgs.msg", nav_msg)

    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.UInt8 = type("UInt8", (), {})
    std_pkg = types.ModuleType("std_msgs")
    std_pkg.msg = std_msg
    sys.modules.setdefault("std_msgs", std_pkg)
    sys.modules.setdefault("std_msgs.msg", std_msg)


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install_ros_import_stubs()
tracker_module = load_module(
    "go2_polar_goal_tracker",
    "ranger_ws/src/go_nav/scripts/polar_goal_tracker.py",
)
bridge_module = load_module(
    "go2_sport_mode_state_to_odom",
    "go2/sport_mode_state_to_odom.py",
)


class Go2ShortTermCompensationTest(unittest.TestCase):
    def make_tracker(self):
        tracker = tracker_module.SimplePolarGoalTracker.__new__(
            tracker_module.SimplePolarGoalTracker
        )
        tracker.robot_x = 0.0
        tracker.robot_y = 0.0
        tracker.robot_yaw = 0.0
        tracker.have_tag_estimate = False
        tracker.tag_x_odom = 0.0
        tracker.tag_y_odom = 0.0
        tracker.alpha = 0.5
        tracker.distance_offset = 0.6
        tracker.min_distance = 0.0
        tracker.theta_offset_rad = 0.0
        tracker.enable_theta_offset = True
        tracker.enable_theta_deadband = False
        tracker.theta_deadband_rad = 0.0
        tracker.camera_offset_x_m = 0.0
        tracker.camera_offset_y_m = 0.0
        return tracker

    def make_timer_tracker(self):
        tracker = self.make_tracker()
        time_cls = tracker_module.rospy.Time
        tracker.state_lock = threading.RLock()
        tracker.use_odom_fallback = True
        tracker.have_odom = True
        tracker.have_tag_estimate = True
        tracker.last_tag_id = 0
        tracker.last_detection_time = time_cls(101.0)
        tracker.last_detection_receive_time = time_cls(1.0)
        tracker.last_detection_array_stamp = time_cls(101.0)
        tracker.last_detection_array_receive_time = time_cls(1.0)
        tracker.last_odom_receive_time = time_cls(1.0)
        tracker.lost_timeout_sec = 0.12
        tracker.predict_timeout_sec = 6.0
        tracker.search_enabled = True
        tracker.search_timeout_sec = 12.0
        tracker.detection_stream_timeout_sec = 0.5
        tracker.odom_timeout_sec = 0.25
        tracker.odom_topic = "/go2/sport_odom"
        tracker.states = []
        tracker.polar_predictions = []
        tracker.publish_tracking_state = tracker.states.append
        tracker.publish_polar = lambda *args: tracker.polar_predictions.append(
            args
        )
        return tracker

    def test_unitree_quaternion_is_wxyz_and_normalized(self):
        root_half = math.sqrt(0.5)
        qw, qx, qy, qz = bridge_module.normalize_quaternion_wxyz(
            [2.0 * root_half, 0.0, 0.0, 2.0 * root_half]
        )
        self.assertAlmostEqual(qw, root_half)
        self.assertAlmostEqual(qz, root_half)
        self.assertAlmostEqual(
            bridge_module.quaternion_to_yaw(qw, qx, qy, qz),
            math.pi / 2.0,
        )

    def test_prediction_tracks_robot_translation_without_double_offset(self):
        tracker = self.make_tracker()
        tracker.update_tag_estimate_from_polar(2.0, 0.0, reset=True)

        self.assertAlmostEqual(tracker.tag_x_odom, 2.0)
        tracker.robot_x = 0.5
        distance, theta = tracker.estimate_polar_from_odom()

        self.assertAlmostEqual(distance, 0.9)
        self.assertAlmostEqual(theta, 0.0)

    def test_prediction_tracks_robot_yaw(self):
        tracker = self.make_tracker()
        tracker.update_tag_estimate_from_polar(2.0, 0.0, reset=True)
        tracker.robot_yaw = math.pi / 2.0

        distance, theta = tracker.estimate_polar_from_odom()

        self.assertAlmostEqual(distance, 1.4)
        self.assertAlmostEqual(theta, -math.pi / 2.0)

    def test_visual_measurement_corrects_cached_tag_position(self):
        tracker = self.make_tracker()
        tracker.alpha = 0.25
        tracker.update_tag_estimate_from_polar(2.0, 0.0, reset=True)
        tracker.update_tag_estimate_from_polar(4.0, 0.0)

        self.assertAlmostEqual(tracker.tag_x_odom, 2.5)
        self.assertAlmostEqual(tracker.tag_y_odom, 0.0)

    def test_camera_translation_is_applied_in_base_frame(self):
        tracker = self.make_tracker()
        tracker.camera_offset_x_m = 0.2
        tracker.camera_offset_y_m = 0.1
        position = types.SimpleNamespace(x=0.0, z=2.0)
        det = types.SimpleNamespace(
            pose=types.SimpleNamespace(
                pose=types.SimpleNamespace(
                    pose=types.SimpleNamespace(position=position)
                )
            )
        )

        distance, theta, _, _, raw_r, theta_for_estimate = (
            tracker.compute_polar(det)
        )

        expected_raw_r = math.hypot(2.2, 0.1)
        expected_theta = math.atan2(0.1, 2.2)
        self.assertAlmostEqual(raw_r, expected_raw_r)
        self.assertAlmostEqual(distance, expected_raw_r - 0.6)
        self.assertAlmostEqual(theta, expected_theta)
        self.assertAlmostEqual(theta_for_estimate, expected_theta)

    def test_empty_detection_array_updates_camera_clock_heartbeat(self):
        tracker = self.make_tracker()
        tracker.state_lock = threading.RLock()
        tracker.last_detection_array_stamp = tracker_module.rospy.Time(0)
        tracker.last_detection_array_receive_time = tracker_module.rospy.Time(0)
        tracker_module.rospy.Time.current_sec = 50.0
        msg = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=tracker_module.rospy.Time(123.0)),
            detections=[],
        )

        tracker.detection_cb(msg)

        self.assertAlmostEqual(tracker.last_detection_array_stamp.to_sec(), 123.0)
        self.assertAlmostEqual(
            tracker.last_detection_array_receive_time.to_sec(), 50.0
        )

    def test_prediction_stamp_advances_on_camera_time_axis(self):
        tracker = self.make_tracker()
        tracker.last_detection_array_stamp = tracker_module.rospy.Time(100.0)
        tracker.last_detection_array_receive_time = tracker_module.rospy.Time(10.0)

        stamp = tracker.prediction_stamp_from_camera(
            tracker_module.rospy.Time(10.25)
        )

        self.assertAlmostEqual(stamp.to_sec(), 100.25)

    def test_tracking_state_values_are_stable(self):
        self.assertEqual(tracker_module.TRACKING_NOT_READY, 0)
        self.assertEqual(tracker_module.TRACKING_VISIBLE, 1)
        self.assertEqual(tracker_module.TRACKING_PREDICTING, 2)
        self.assertEqual(tracker_module.TRACKING_SEARCHABLE, 3)

    def test_timer_transitions_from_prediction_to_bounded_search_and_stop(self):
        tracker = self.make_timer_tracker()
        time_cls = tracker_module.rospy.Time

        time_cls.current_sec = 5.0
        tracker.last_odom_receive_time = time_cls(5.0)
        tracker.last_detection_array_receive_time = time_cls(5.0)
        tracker.last_detection_array_stamp = time_cls(105.0)
        tracker.timer_cb(None)
        self.assertEqual(tracker.states[-1], tracker_module.TRACKING_PREDICTING)
        self.assertEqual(len(tracker.polar_predictions), 1)

        time_cls.current_sec = 8.0
        tracker.last_odom_receive_time = time_cls(8.0)
        tracker.last_detection_array_receive_time = time_cls(8.0)
        tracker.last_detection_array_stamp = time_cls(108.0)
        tracker.timer_cb(None)
        self.assertEqual(tracker.states[-1], tracker_module.TRACKING_SEARCHABLE)
        self.assertEqual(len(tracker.polar_predictions), 1)

        time_cls.current_sec = 20.0
        tracker.last_odom_receive_time = time_cls(20.0)
        tracker.last_detection_array_receive_time = time_cls(20.0)
        tracker.timer_cb(None)
        self.assertEqual(tracker.states[-1], tracker_module.TRACKING_NOT_READY)

    def test_stale_detection_stream_never_requests_search(self):
        tracker = self.make_timer_tracker()
        time_cls = tracker_module.rospy.Time
        time_cls.current_sec = 8.0
        tracker.last_odom_receive_time = time_cls(8.0)
        tracker.last_detection_array_receive_time = time_cls(7.4)

        tracker.timer_cb(None)

        self.assertEqual(tracker.states[-1], tracker_module.TRACKING_NOT_READY)
        self.assertEqual(tracker.polar_predictions, [])


if __name__ == "__main__":
    unittest.main()
