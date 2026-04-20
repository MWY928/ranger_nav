#!/home/mobile/miniconda3/envs/simple_nav/bin/python
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import PointStamped, Twist


class TagFollowerNode(object):
    def __init__(self):
        rospy.init_node("tag_follower_node", anonymous=False)

        self.input_topic = rospy.get_param("~input_topic", "/tag_polar")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        self.target_distance = float(rospy.get_param("~target_distance", 1.0))
        self.k_r = float(rospy.get_param("~k_r", 0.6))
        self.k_theta = float(rospy.get_param("~k_theta", 1.2))

        self.max_linear = float(rospy.get_param("~max_linear", 1.0))
        self.max_angular = float(rospy.get_param("~max_angular", 1.0))

        self.angle_deadband = float(rospy.get_param("~angle_deadband", 0.05))
        self.dist_deadband = float(rospy.get_param("~dist_deadband", 0.03))

        self.search_timeout = float(rospy.get_param("~search_timeout", 0.5))
        self.forward_only_when_facing = bool(rospy.get_param("~forward_only_when_facing", True))
        self.facing_angle_thresh = float(rospy.get_param("~facing_angle_thresh", 0.35))

        self.last_msg_time = None

        self.pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.sub = rospy.Subscriber(self.input_topic, PointStamped, self.cb, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(0.05), self.watchdog)

        rospy.loginfo("tag_follower_node started")
        rospy.loginfo("Subscribe: %s", self.input_topic)
        rospy.loginfo("Publish:   %s", self.cmd_vel_topic)

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def stop(self):
        self.pub.publish(Twist())

    def cb(self, msg):
        self.last_msg_time = rospy.Time.now()

        r = msg.point.x
        theta = msg.point.y
        tag_id = int(msg.point.z)

        dist_err = r - self.target_distance

        cmd = Twist()

        # 角度小死区
        if abs(theta) < self.angle_deadband:
            theta = 0.0

        # 距离小死区
        if abs(dist_err) < self.dist_deadband:
            dist_err = 0.0

        # 角速度控制
        cmd.angular.z = self.clamp(self.k_theta * theta,
                                   -self.max_angular, self.max_angular)

        # 线速度控制
        if dist_err > 0.0:
            if self.forward_only_when_facing and abs(theta) > self.facing_angle_thresh:
                cmd.linear.x = 0.0
            else:
                cmd.linear.x = self.clamp(self.k_r * dist_err,
                                          0.0, self.max_linear)
        else:
            cmd.linear.x = 0.0

        self.pub.publish(cmd)

        rospy.loginfo_throttle(
            0.5,
            "tag=%d r=%.3f theta=%.3f -> vx=%.3f wz=%.3f",
            tag_id, r, theta, cmd.linear.x, cmd.angular.z
        )

    def watchdog(self, _event):
        if self.last_msg_time is None:
            self.stop()
            return

        dt = (rospy.Time.now() - self.last_msg_time).to_sec()
        if dt > self.search_timeout:
            self.stop()


if __name__ == "__main__":
    try:
        TagFollowerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
