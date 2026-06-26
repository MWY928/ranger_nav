#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import rospy
from std_msgs.msg import Int32


ACTION_NAMES = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}


class UnitreeSportController(object):
    def __init__(self, args):
        self.dry_run = bool(args.dry_run)
        self.network_interface = args.network_interface
        self.domain_id = int(args.domain_id)
        self.timeout_sec = float(args.sdk_timeout_sec)
        self.client = None

        if self.dry_run:
            rospy.logwarn("Unitree dry-run enabled; SDK2 commands are only logged.")
            return

        if not self.network_interface:
            raise RuntimeError(
                "--network_interface is required unless --dry_run is enabled."
            )

        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
        except ImportError as exc:
            raise RuntimeError(
                "unitree_sdk2py is not importable in this environment."
            ) from exc

        ChannelFactoryInitialize(self.domain_id, self.network_interface)
        self.client = SportClient()
        self.client.SetTimeout(self.timeout_sec)
        self.client.Init()

        if args.balance_stand_on_start:
            self._call("BalanceStand")

        rospy.loginfo(
            "Unitree SportClient ready: interface=%s domain_id=%d timeout=%.2fs",
            self.network_interface,
            self.domain_id,
            self.timeout_sec,
        )

    def _call(self, method_name, *args):
        if self.client is None:
            return None
        result = getattr(self.client, method_name)(*args)
        if result not in (None, 0):
            rospy.logwarn_throttle(
                1.0,
                "Unitree %s returned non-zero result: %s",
                method_name,
                str(result),
            )
        return result

    def move(self, vx, vy, vyaw):
        if self.dry_run:
            rospy.loginfo_throttle(
                0.5,
                "[UNITREE_DRY_RUN] Move(vx=%.3f, vy=%.3f, vyaw=%.3f)",
                vx,
                vy,
                vyaw,
            )
            return
        self._call("Move", float(vx), float(vy), float(vyaw))

    def stop(self):
        if self.dry_run:
            rospy.loginfo_throttle(0.5, "[UNITREE_DRY_RUN] StopMove()")
            return
        self._call("StopMove")


class UnitreeActionMapper(object):
    def __init__(self, args):
        rospy.init_node("unitree_action_mapper", anonymous=False)
        self.args = args
        self.controller = UnitreeSportController(args)

        self.action_to_velocity = {
            0: (0.0, 0.0, 0.0),
            1: (args.forward_speed, 0.0, 0.0),
            2: (0.0, 0.0, args.turn_speed),
            3: (0.0, 0.0, -args.turn_speed),
        }
        self.last_action_time = rospy.Time(0)
        self.last_action_id = None
        self.stopped_for_timeout = False

        self.sub = rospy.Subscriber(
            args.action_topic,
            Int32,
            self.action_cb,
            queue_size=1,
        )
        self.watchdog = rospy.Timer(
            rospy.Duration(1.0 / max(args.watchdog_rate_hz, 1.0)),
            self.watchdog_cb,
        )
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("Unitree action mapper started.")
        rospy.loginfo("Subscribe action_topic: %s", args.action_topic)
        rospy.loginfo(
            "Mapping: 0 stop, 1 Move(%.3f,0,0), 2 Move(0,0,%.3f), 3 Move(0,0,-%.3f)",
            args.forward_speed,
            args.turn_speed,
            args.turn_speed,
        )
        rospy.loginfo("action_timeout_sec: %.3f", args.action_timeout_sec)

    def action_cb(self, msg):
        action_id = int(msg.data)
        self.last_action_time = rospy.Time.now()
        self.last_action_id = action_id
        self.stopped_for_timeout = False

        if action_id not in self.action_to_velocity:
            rospy.logwarn_throttle(1.0, "Unknown action_id=%d; stopping.", action_id)
            self.controller.stop()
            return

        vx, vy, vyaw = self.action_to_velocity[action_id]
        action_name = ACTION_NAMES.get(action_id, "action_{}".format(action_id))
        rospy.loginfo_throttle(
            0.5,
            "action_id=%d(%s) -> vx=%.3f vy=%.3f vyaw=%.3f",
            action_id,
            action_name,
            vx,
            vy,
            vyaw,
        )

        if action_id == 0:
            self.controller.stop()
            return
        self.controller.move(vx, vy, vyaw)

    def watchdog_cb(self, _event):
        if self.last_action_time == rospy.Time(0):
            return

        age = (rospy.Time.now() - self.last_action_time).to_sec()
        if age <= self.args.action_timeout_sec:
            return

        if not self.stopped_for_timeout:
            rospy.logwarn(
                "No action for %.3fs > %.3fs; stopping Go2.",
                age,
                self.args.action_timeout_sec,
            )
            self.controller.stop()
            self.stopped_for_timeout = True

    def shutdown(self):
        if self.args.stop_on_shutdown:
            try:
                self.controller.stop()
            except Exception as exc:
                rospy.logerr("Shutdown stop failed: %s", str(exc))

    def spin(self):
        rospy.spin()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map Falcon discrete ROS actions to Unitree SDK2 Go2 commands."
    )
    parser.add_argument("--action_topic", type=str, default="/falcon/action_id")
    parser.add_argument("--network_interface", type=str, default="")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--sdk_timeout_sec", type=float, default=10.0)
    parser.add_argument("--forward_speed", type=float, default=0.6)
    parser.add_argument("--turn_speed", type=float, default=0.6)
    parser.add_argument("--action_timeout_sec", type=float, default=0.3)
    parser.add_argument("--watchdog_rate_hz", type=float, default=20.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--balance_stand_on_start", action="store_true")
    parser.add_argument("--no_stop_on_shutdown", dest="stop_on_shutdown", action="store_false")
    parser.set_defaults(stop_on_shutdown=True)
    return parser.parse_args(rospy.myargv()[1:])


if __name__ == "__main__":
    node = UnitreeActionMapper(parse_args())
    node.spin()
