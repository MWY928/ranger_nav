import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def install_ros_import_stubs():
    rospy = types.ModuleType("rospy")
    rospy.Time = type("Time", (), {})
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


if __name__ == "__main__":
    unittest.main()
