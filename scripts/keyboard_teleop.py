#!/usr/bin/env python3
"""Keyboard teleop for the CMU-VLN mecanum robot.

Publishes sensor_msgs/msg/Joy on /joy with the mapping expected by the
autonomy stack's pathFollower (see local_planner/src/pathFollower.cpp):

    axes[0] -> manual yaw   (rotate)
    axes[2] -> autonomy trigger (kept at +1.0 => autonomy OFF)
    axes[3] -> manual strafe (linear.y, needs omniDirGoalThre > 0)
    axes[4] -> manual forward (linear.x)
    axes[5] -> manual trigger (kept at -1.0 => manual mode ON)

In manual mode the pathFollower drives the base directly from these axes,
so this gives full holonomic keyboard control without a physical joystick.

Controls (persistent "cruise" model, like teleop_twist_keyboard):
    w / s : forward / backward
    a / d : strafe left / right
    q / e : rotate left / right
    space or k : stop (zero all motion)
    z / x : decrease / increase speed scale
    Ctrl-C : quit (sends a stop first)
"""

import sys
import select
import termios
import time
import tty
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

HELP = """
CMU-VLN keyboard teleop (manual mode via /joy)
------------------------------------------------
  w/s : forward / back      a/d : strafe left / right
  q/e : rotate left / right
  space or k : STOP
  z/x : speed -/+           Ctrl-C : quit
------------------------------------------------
"""

# Axis magnitudes (fraction of maxSpeed / maxYawRate) applied per key.
STEP_LIN = 0.6
STEP_YAW = 0.6
PUBLISH_HZ = 50.0


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("keyboard_teleop")
        self.pub = self.create_publisher(Joy, "/joy", 10)
        self.fwd = 0.0
        self.strafe = 0.0
        self.yaw = 0.0
        self.scale = 1.0
        self.lock = threading.Lock()
        self.create_timer(1.0 / PUBLISH_HZ, self._publish)

    def set_cmd(self, fwd, strafe, yaw):
        with self.lock:
            self.fwd, self.strafe, self.yaw = fwd, strafe, yaw

    def set_scale(self, scale):
        with self.lock:
            self.scale = max(0.1, min(1.0, scale))

    def _publish(self):
        with self.lock:
            fwd, strafe, yaw, scale = self.fwd, self.strafe, self.yaw, self.scale
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        # 8-axis / 11-button layout matching a standard gamepad.
        msg.axes = [0.0] * 8
        msg.buttons = [0] * 11
        msg.axes[0] = _clamp(yaw * scale)      # manual yaw
        msg.axes[2] = 1.0                       # autonomy OFF (> -0.1)
        msg.axes[3] = _clamp(strafe * scale)   # manual strafe (left +)
        msg.axes[4] = _clamp(fwd * scale)      # manual forward (+)
        msg.axes[5] = -1.0                      # manual mode ON (<= -0.1)
        self.pub.publish(msg)

    def publish_release(self):
        """Hand control back to the autonomy stack on exit.

        manualMode OFF (axes[5] > -0.1) and autonomyMode ON (axes[2] <= -0.1)
        so waypoint following works again after teleop quits.
        """
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [0.0] * 8
        msg.buttons = [0] * 11
        msg.axes[2] = -1.0   # autonomy ON
        msg.axes[5] = 1.0    # manual OFF
        self.pub.publish(msg)


def _clamp(v):
    return max(-1.0, min(1.0, v))


def main():
    rclpy.init()
    node = KeyboardTeleop()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(HELP)
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while rclpy.ok():
            key = _get_key(settings)
            if key is None:
                continue
            if key == "\x03":  # Ctrl-C
                break
            elif key == "w":
                node.set_cmd(STEP_LIN, 0.0, 0.0)
            elif key == "s":
                node.set_cmd(-STEP_LIN, 0.0, 0.0)
            elif key == "a":
                node.set_cmd(0.0, STEP_LIN, 0.0)   # strafe left
            elif key == "d":
                node.set_cmd(0.0, -STEP_LIN, 0.0)  # strafe right
            elif key == "q":
                node.set_cmd(0.0, 0.0, STEP_YAW)   # rotate left (CCW)
            elif key == "e":
                node.set_cmd(0.0, 0.0, -STEP_YAW)  # rotate right (CW)
            elif key in (" ", "k"):
                node.set_cmd(0.0, 0.0, 0.0)
            elif key == "x":
                node.set_scale(node.scale + 0.1)
                _status(node)
            elif key == "z":
                node.set_scale(node.scale - 0.1)
                _status(node)
    finally:
        # Halt, then hand control back to autonomy so waypoints work again.
        node.set_cmd(0.0, 0.0, 0.0)
        for _ in range(5):
            node._publish()
            time.sleep(0.02)
        for _ in range(5):
            node.publish_release()
            time.sleep(0.02)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()
        print("\r\nteleop stopped.\r")


def _status(node):
    sys.stdout.write("\rspeed scale: %.1f   \r" % node.scale)
    sys.stdout.flush()


def _get_key(settings, timeout=0.1):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.read(1)
    return None


if __name__ == "__main__":
    main()
