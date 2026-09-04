#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import threading

import rospy
from std_msgs.msg import Int32


ACTION_NAMES = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
    4: "search_left",
    5: "search_right",
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value, got {!r}".format(value)
    )


class VelocitySlewLimiter(object):
    """Move current velocity toward a target using separate accel/decel limits."""

    def __init__(
        self,
        linear_accel_limit,
        linear_decel_limit,
        yaw_accel_limit,
        yaw_decel_limit,
    ):
        self.linear_accel_limit = self._positive(
            "linear_accel_limit", linear_accel_limit
        )
        self.linear_decel_limit = self._positive(
            "linear_decel_limit", linear_decel_limit
        )
        self.yaw_accel_limit = self._positive(
            "yaw_accel_limit", yaw_accel_limit
        )
        self.yaw_decel_limit = self._positive(
            "yaw_decel_limit", yaw_decel_limit
        )
        self.current = (0.0, 0.0, 0.0)
        self.target = (0.0, 0.0, 0.0)

    @staticmethod
    def _positive(name, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
        return value

    @staticmethod
    def _approach(current, target, accel_limit, decel_limit, dt):
        if current * target < 0.0:
            # Direction changes must brake to zero before accelerating oppositely.
            target_for_step = 0.0
            rate_limit = decel_limit
        else:
            target_for_step = target
            rate_limit = (
                accel_limit if abs(target) > abs(current) else decel_limit
            )

        max_delta = rate_limit * dt
        delta = target_for_step - current
        if abs(delta) <= max_delta:
            return target_for_step
        return current + math.copysign(max_delta, delta)

    def set_target(self, vx, vy, vyaw):
        self.target = (float(vx), float(vy), float(vyaw))

    def reset(self):
        self.current = (0.0, 0.0, 0.0)
        self.target = (0.0, 0.0, 0.0)

    def step(self, dt):
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")

        vx = self._approach(
            self.current[0],
            self.target[0],
            self.linear_accel_limit,
            self.linear_decel_limit,
            dt,
        )
        vy = self._approach(
            self.current[1],
            self.target[1],
            self.linear_accel_limit,
            self.linear_decel_limit,
            dt,
        )
        vyaw = self._approach(
            self.current[2],
            self.target[2],
            self.yaw_accel_limit,
            self.yaw_decel_limit,
            dt,
        )
        self.current = (vx, vy, vyaw)
        return self.current


class UnitreeSportController(object):
    def __init__(self, args):
        self.dry_run = bool(args.dry_run)
        self.network_interface = args.network_interface
        self.domain_id = int(args.domain_id)
        self.timeout_sec = float(args.sdk_timeout_sec)
        self.client = None

        if self.dry_run:
            rospy.logwarn(
                "Unitree dry-run enabled; SDK2 commands are only logged."
            )
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
            4: (0.0, 0.0, args.search_turn_speed),
            5: (0.0, 0.0, -args.search_turn_speed),
        }
        self.last_action_time = rospy.Time(0)
        self.last_action_id = None
        self.stopped_for_timeout = False
        self.state_lock = threading.RLock()
        # Never hold state_lock while calling the SDK: a network RPC may block.
        # SDK calls are serialized separately, and command_generation invalidates
        # Move calls that were prepared before a safety stop or action change.
        self.sdk_call_lock = threading.Lock()
        self.command_generation = 0
        self.pending_stop_calls = 0
        self.shutdown_requested = False
        self.smoothing_enabled = bool(args.smoothing_enabled)
        self.velocity_limiter = VelocitySlewLimiter(
            linear_accel_limit=args.linear_accel_limit,
            linear_decel_limit=args.linear_decel_limit,
            yaw_accel_limit=args.yaw_accel_limit,
            yaw_decel_limit=args.yaw_decel_limit,
        )
        self.last_control_time = rospy.Time.now()

        self.sub = rospy.Subscriber(
            args.action_topic,
            Int32,
            self.action_cb,
            queue_size=1,
        )
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / max(args.watchdog_rate_hz, 1.0)),
            self.control_cb,
        )
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("Unitree action mapper started.")
        rospy.loginfo("Subscribe action_topic: %s", args.action_topic)
        rospy.loginfo(
            "Mapping: 0 stop, 1 Move(%.3f,0,0), "
            "2 Move(0,0,%.3f), 3 Move(0,0,-%.3f), "
            "4 search-left Move(0,0,%.3f), "
            "5 search-right Move(0,0,-%.3f)",
            args.forward_speed,
            args.turn_speed,
            args.turn_speed,
            args.search_turn_speed,
            args.search_turn_speed,
        )
        rospy.loginfo("action_timeout_sec: %.3f", args.action_timeout_sec)
        rospy.loginfo(
            "Velocity smoothing: enabled=%s rate=%.1fHz "
            "linear_accel/decel=%.3f/%.3f yaw_accel/decel=%.3f/%.3f",
            self.smoothing_enabled,
            args.watchdog_rate_hz,
            args.linear_accel_limit,
            args.linear_decel_limit,
            args.yaw_accel_limit,
            args.yaw_decel_limit,
        )

    def _prepare_stop_locked(self):
        """Reset motion state and create a barrier for already queued Move calls."""
        self.command_generation += 1
        self.pending_stop_calls += 1
        self.velocity_limiter.reset()

    def _execute_stop(self):
        """Serialize StopMove without holding state_lock."""
        try:
            with self.sdk_call_lock:
                self.controller.stop()
        finally:
            with self.state_lock:
                self.pending_stop_calls -= 1

    def _execute_move(self, move_generation, velocity):
        """Execute a Move only if it is still current after SDK serialization."""
        with self.sdk_call_lock:
            with self.state_lock:
                move_is_current = (
                    move_generation == self.command_generation
                    and self.pending_stop_calls == 0
                    and not self.shutdown_requested
                    and not self.stopped_for_timeout
                    and self.last_action_id in self.action_to_velocity
                    and self.last_action_id != 0
                )
            if not move_is_current:
                return False

            # A stop request can now update state, but it must wait for this RPC.
            # It will therefore execute after this Move and remain the final command.
            self.controller.move(*velocity)
            return True

    def action_cb(self, msg):
        action_id = int(msg.data)
        should_stop = False
        with self.state_lock:
            if self.shutdown_requested:
                return

            previous_action_id = self.last_action_id
            was_stopped_for_timeout = self.stopped_for_timeout
            self.last_action_time = rospy.Time.now()
            self.last_action_id = action_id
            self.stopped_for_timeout = False

            if action_id not in self.action_to_velocity:
                rospy.logwarn_throttle(
                    1.0, "Unknown action_id=%d; stopping.", action_id
                )
                self._prepare_stop_locked()
                should_stop = True
            else:
                vx, vy, vyaw = self.action_to_velocity[action_id]
                action_name = ACTION_NAMES.get(
                    action_id, "action_{}".format(action_id)
                )
                rospy.loginfo_throttle(
                    0.5,
                    "action_id=%d(%s) -> target vx=%.3f vy=%.3f vyaw=%.3f",
                    action_id,
                    action_name,
                    vx,
                    vy,
                    vyaw,
                )

            if action_id == 0:
                # Action 0 may be a bridge safety stop, so never delay it with a ramp.
                self._prepare_stop_locked()
                should_stop = True
            elif action_id in self.action_to_velocity:
                if action_id != previous_action_id or was_stopped_for_timeout:
                    self.command_generation += 1
                self.velocity_limiter.set_target(vx, vy, vyaw)

        if should_stop:
            self._execute_stop()

    def control_cb(self, _event):
        should_stop = False
        move_command = None
        with self.state_lock:
            now = rospy.Time.now()
            elapsed = max(0.0, (now - self.last_control_time).to_sec())
            self.last_control_time = now

            if self.shutdown_requested or self.last_action_time == rospy.Time(
                0
            ):
                return

            age = (now - self.last_action_time).to_sec()
            if age > self.args.action_timeout_sec:
                if not self.stopped_for_timeout:
                    rospy.logwarn(
                        "No action for %.3fs > %.3fs; stopping Go2.",
                        age,
                        self.args.action_timeout_sec,
                    )
                    self.stopped_for_timeout = True
                    self._prepare_stop_locked()
                    should_stop = True
            else:
                if self.last_action_id not in self.action_to_velocity:
                    return
                if self.last_action_id == 0:
                    return
                if self.pending_stop_calls:
                    return

                max_dt = 2.0 / max(self.args.watchdog_rate_hz, 1.0)
                dt = min(elapsed, max_dt)
                if self.smoothing_enabled:
                    vx, vy, vyaw = self.velocity_limiter.step(dt)
                else:
                    vx, vy, vyaw = self.velocity_limiter.target
                    self.velocity_limiter.current = (vx, vy, vyaw)
                move_command = (
                    self.command_generation,
                    (vx, vy, vyaw),
                )

        if should_stop:
            self._execute_stop()
            return
        if move_command is not None and self._execute_move(*move_command):
            vx, vy, vyaw = move_command[1]
            rospy.logdebug(
                "smoothed velocity vx=%.3f vy=%.3f vyaw=%.3f",
                vx,
                vy,
                vyaw,
            )

    def shutdown(self):
        if self.args.stop_on_shutdown:
            try:
                with self.state_lock:
                    self.shutdown_requested = True
                    self._prepare_stop_locked()
                self._execute_stop()
            except Exception as exc:
                rospy.logerr("Shutdown stop failed: %s", str(exc))

    def spin(self):
        rospy.spin()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map Falcon discrete ROS actions to Unitree SDK2 Go2 commands."
    )
    parser.add_argument(
        "--action_topic", type=str, default="/falcon/action_id"
    )
    parser.add_argument("--network_interface", type=str, default="")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--sdk_timeout_sec", type=float, default=10.0)
    parser.add_argument("--forward_speed", type=float, default=0.6)
    parser.add_argument("--turn_speed", type=float, default=0.6)
    parser.add_argument("--search_turn_speed", type=float, default=0.25)
    parser.add_argument("--action_timeout_sec", type=float, default=0.3)
    parser.add_argument("--watchdog_rate_hz", type=float, default=20.0)
    parser.add_argument(
        "--smoothing_enabled",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--linear_accel_limit", type=float, default=1.0)
    parser.add_argument("--linear_decel_limit", type=float, default=1.5)
    parser.add_argument("--yaw_accel_limit", type=float, default=2.0)
    parser.add_argument("--yaw_decel_limit", type=float, default=3.0)
    parser.add_argument(
        "--dry_run", type=str_to_bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--balance_stand_on_start",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--no_stop_on_shutdown", dest="stop_on_shutdown", action="store_false"
    )
    parser.set_defaults(stop_on_shutdown=True)
    return parser.parse_args(rospy.myargv()[1:])


if __name__ == "__main__":
    node = UnitreeActionMapper(parse_args())
    node.spin()
