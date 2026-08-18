"""Shared bootstrap + worker-thread plumbing for sam_node and map_node.

Both nodes (docs/M2_perception.md 3.6-split) run a background worker thread pulling off
a drop-to-latest buffer, track a `stage`/`stage_since` pair for heartbeat reporting, and
are launched the same way (ros2 launch -> config param/env var -> yaml -> read
`runtime.executor_threads` -> MultiThreadedExecutor). This module is the only place that
plumbing lives; the two nodes still define their own `_worker_loop`/`_heartbeat` bodies
since what happens per frame is genuinely different.
"""
from __future__ import annotations

import contextlib
import os
import signal
import threading
import time

import rclpy
import yaml
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node


def load_config(path: str, required_keys: tuple) -> dict:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    for key in required_keys:
        if key not in config:
            raise KeyError(f"config {path} is missing required key '{key}'")
    return config


def resolve_config_path(config: str) -> str:
    """A bare filename resolves against the package config dir; an absolute path is used
    as given — the same rule sam_node.launch / map_node.launch apply.

    Exists so the offline tools (record_companion, replay_map3d) load the SAME yaml the
    nodes do. They previously defaulted to an empty dict, which silently gave SAM 3 its
    library defaults instead of the tuned image_size and detection thresholds — i.e. the
    recorded companion bag would have come from a different detector than production.
    """
    if os.path.isabs(config):
        return config
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("sam_mapper")
    except Exception:
        share = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(share, "config", config)


def run_node(node_cls, bootstrap_name: str, required_keys: tuple, launch_hint: str, args=None) -> None:
    """Standard main(): resolve config path (ros2 param or env var) -> load -> spin node_cls(config)."""
    rclpy.init(args=args)

    tmp = Node(bootstrap_name)
    tmp.declare_parameter('config', '')
    config_path = tmp.get_parameter('config').get_parameter_value().string_value
    tmp.destroy_node()

    if not config_path:
        config_path = os.environ.get('SAM_MAPPER_CONFIG', '')
    if not config_path or not os.path.isfile(config_path):
        raise SystemExit(
            f"config not found: '{config_path}'. Pass -p config:=/path/to/sam3_mecanum_sim.yaml "
            f"or use: {launch_hint}"
        )

    config = load_config(config_path, required_keys)
    node = node_cls(config)

    threads = int(config.get('runtime', {}).get('executor_threads', 8))
    executor = MultiThreadedExecutor(num_threads=threads)
    node.log(f'executor threads: {threads}')
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        # SIGINT arrives twice on a launch teardown — the terminal delivers it to the
        # whole foreground process group, and `ros2 launch` also signals each child — so
        # without this the second one lands mid-destroy_node() and escapes as a
        # KeyboardInterrupt traceback out of rclpy's cleanup, on an ordinary Ctrl-C.
        # (captioner.ros_utils.shutdown_guard is the same guard; duplicated rather than
        # imported because sam_mapper does not depend on captioner and four lines is a
        # poor reason to make it.) Not restored: the process is exiting, and a restored
        # handler could let a queued signal fire right after the guard.
        with contextlib.suppress(ValueError):  # not the main thread: nothing to install
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.destroy_node()
            # executor.spin() already tears the context down on SIGINT; calling shutdown
            # a second time raises RCLError and turns a clean Ctrl-C into exit code 1,
            # which makes the eval harness's teardown look like a crash.
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


class WorkerNodeMixin:
    """Mixin for a Node that runs a background worker thread and reports its stage.

    Subclasses call `self._start_worker(self._worker_loop)` once their ROS interface is
    set up, and define `_worker_loop(self)` themselves.
    """

    @staticmethod
    def _stamp_of(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

    def _start_worker(self, loop_fn) -> None:
        self.stage = 'starting'
        self.stage_since = time.monotonic()
        self.running = True
        self.worker = threading.Thread(target=loop_fn, daemon=True)
        self.worker.start()

    def _set_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_since = time.monotonic()

    def destroy_node(self):
        self.running = False
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)
        super().destroy_node()
