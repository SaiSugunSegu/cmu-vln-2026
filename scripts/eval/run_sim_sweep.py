#!/usr/bin/env python3
"""Live-sim eval sweep. Runs on the HOST, because only the host can change scenes.

`eval_orchestrator` runs inside iros2026_odyssey. It can walk every scene of a BAG
sweep by itself, since bags are just files under /data/bags that it can open. It cannot
do the same for the simulator: the Unity mesh is a single overwritable slot inside
iros2026_system's image, and that container mounts nothing that would let the AI module
reach it. Swapping a scene therefore means `docker cp` plus a sim restart, which only a
process outside both containers can do.

So this script owns the outer loop and calls the orchestrator once per scene:

    for scene in scenes:
        stop the sim
        docker cp <scenes-dir>/<scene>/  ->  the system container's mesh slot
        start challenge_simulation.sh --noviz   (domain 42 + topic firewall)
        wait until /state_estimation, /camera/image and /registered_scan
            have publishers in domain 0
        eval_orchestrator scene_source:=sim scene:=<scene>   (relaunches per question)
        stop the sim
    merge the per-scene reports

The robot is NOT returned to a start pose between questions. Every question relaunches
the pipeline and rebuilds the 3D map from scratch, so it has to re-explore for its own
targets regardless of where the previous one left the robot.

Usage:
  scripts/eval/run_sim_sweep.py --category 1 --scenes arabic_room chinese_room
  scripts/eval/run_sim_sweep.py --category 2 --scenes all --limit 2
  scripts/eval/run_sim_sweep.py --category 3 --scenes arabic_room
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The report is written by the orchestrator inside the container; we only read it back to
# print the sweep total. Reusing its own helpers means there is no second summary
# implementation to drift. report_utils is stdlib-only (no rclpy) — which is what makes it
# importable out here on the host, unlike eval_orchestrator itself.
sys.path.insert(0, str(REPO / "ai_module" / "src" / "smart_vlm"))
from smart_vlm.report_utils import previous_results, summarise  # noqa: E402

SYS_CONTAINER = "iros2026_system"
AI_CONTAINER = "iros2026_odyssey"
STACK = "/home/docker/autonomy_stack_mecanum_wheel_platform"
MESH_SLOT = f"{STACK}/src/base_autonomy/vehicle_simulator/mesh/unity"
AI_SRC = "/home/docker/ai_module"

# Everything the sim half starts, including the domain bridge — a leftover bridge would
# relay the next scene's topics from a simulator that is no longer running.
SIM_PROCS = "challenge_simulation|system_simulation|Model.x86_64|domain_bridge|vehicleSimulator"


def host_path(container_path: str) -> Path:
    """Translate a container path under /data to where it actually is on the host.

    This script straddles both sides: the orchestrator runs inside the AI container and
    must be given /data/..., while the merge at the end reads and writes the same files
    from out here. docker/compose_gpu.yml bind-mounts <repo>/data to /data, so the mapping
    is a prefix swap.
    """
    if container_path.startswith("/data/"):
        return REPO / "data" / container_path[len("/data/"):]
    return Path(container_path)


def log(msg: str, *, err: bool = False) -> None:
    print(f"[sim-sweep] {msg}", file=sys.stderr if err else sys.stdout, flush=True)


def sh(cmd: list[str], *, check: bool = True, capture: bool = False):
    return subprocess.run(cmd, check=check, text=True,
                          capture_output=capture)


def stop_sim() -> None:
    """Kill the sim half. Never `check`: nothing running is the normal case."""
    sh(["docker", "exec", SYS_CONTAINER, "bash", "-lc",
        f"pkill -9 -f '{SIM_PROCS}' || true"], check=False, capture=True)
    time.sleep(3)


def swap_scene(scene_dir: Path) -> None:
    """Overwrite the mesh slot with this scene.

    `docker cp`, not a host copy: the slot lives in the system container's image and is
    not bind-mounted anywhere. (scripts/eval/run_bench.py copies into the host repo
    instead, which is why it never actually switched scenes.)
    """
    sh(["docker", "exec", SYS_CONTAINER, "bash", "-lc", f"rm -rf {MESH_SLOT}/* || true"],
       check=False, capture=True)
    sh(["docker", "cp", f"{scene_dir}/.", f"{SYS_CONTAINER}:{MESH_SLOT}/"], capture=True)


def resolve_display(requested: str) -> str:
    """Use --display if given, otherwise start Xvfb :99 in the system container.

    This machine has no monitor. Unity still needs a DISPLAY; Xvfb is the dummy
    X, and vglrun -d egl renders on the NVIDIA GPU into it.
    """
    if requested:
        return requested
    out = sh([str(REPO / "scripts" / "eval" / "ensure_xvfb.sh")], capture=True)
    display = out.stdout.strip() or ":99"
    if out.stderr.strip():
        log(out.stderr.strip())
    return display


def start_sim(display: str) -> None:
    sh(["docker", "exec", "-d",
        "-e", f"DISPLAY={display}",
        "-e", "XDG_RUNTIME_DIR=/tmp/runtime-docker",
        SYS_CONTAINER, "bash", "-lc",
        f"mkdir -p /tmp/runtime-docker && chmod 700 /tmp/runtime-docker && "
        f"cd {STACK} && vglrun -d egl ./challenge_simulation.sh --noviz "
        f"> /tmp/challenge_sim.log 2>&1"], capture=True)


#: The three topics that together prove BOTH halves are alive: the base autonomy stack
#: (vehicleSimulator) publishes odometry, the Unity simulator publishes the sensors.
SIM_TOPICS = ("/state_estimation", "/registered_scan", "/camera/image")

#: eval_orchestrator's exit code for "the sim stopped feeding mid-question". Kept in step
#: with smart_vlm.eval_orchestrator.EXIT_SENSORS_LOST -- this script runs on the HOST and
#: cannot import from the container's package.
EXIT_SENSORS_LOST = 3

#: How long to wait for one message per topic. /registered_scan is 5 Hz and /camera/image
#: 10 Hz, so this is ample; three of these back to back also serve as the poll interval.
PROBE_S = 5.0


def _has_data(topic: str) -> bool:
    """Did a real message arrive on `topic`? Not 'does a publisher exist' -- see wait_for_sim.

    `--field header.stamp` keeps the transfer tiny: /camera/image is a 1920x640 frame and we
    only need to know that one arrived.
    """
    out = sh(["docker", "exec", AI_CONTAINER, "bash", "-lc",
              f"source {AI_SRC}/install/setup.bash && "
              f"ros2 topic echo --once --no-arr --timeout {PROBE_S:.0f} "
              f"--field header.stamp {topic} >/dev/null 2>&1"],
             check=False, capture=True)
    return out.returncode == 0


def wait_for_sim(timeout_s: float) -> list[str]:
    """Block until the AI module's domain actually RECEIVES sensor data.

    Returns the topics still silent when it gives up; an empty list means alive.

    The old check counted PUBLISHERS, and a publisher count is not data. `stop_sim` kills the
    sim with SIGKILL, and a killed Fast-DDS participant never announces its departure, so the
    graph keeps listing the dead sim's publishers until the participant lease expires. The
    check was therefore satisfied by the previous sim's ghost: across a 15-scene sweep it
    returned true a uniform 10.8-12.2 s after every restart -- far too fast for Unity to load a
    scene -- including for five scenes whose sensors never came up at all, whose nine questions
    then scored 0 for an infrastructure failure rather than for anything the system did.

    Waiting for a real message cannot be fooled that way: a process that is not running
    publishes nothing.
    """
    deadline = time.monotonic() + timeout_s
    silent = list(SIM_TOPICS)
    while True:
        silent = [t for t in SIM_TOPICS if not _has_data(t)]
        if not silent or time.monotonic() >= deadline:
            return silent


def bring_up_sim(scene: str, scene_dir: Path, display: str,
                 timeout_s: float, attempts: int) -> list[str]:
    """Start the sim for this scene, restarting it if it does not come alive.

    Returns the topics still silent after the last attempt; empty means alive.

    Restarting rather than skipping on the first failure, because the failure is transient:
    in the sweep that exposed this, office_1 produced nothing for its first question and was
    fully alive by its second, Unity having simply come up late. Skipping immediately throws
    away questions a restart would have rescued.
    """
    silent: list[str] = list(SIM_TOPICS)
    for attempt in range(1, max(1, attempts) + 1):
        stop_sim()
        if attempt == 1:
            # Only once: a retry re-runs Unity against a slot that is already correct, so it
            # costs a restart rather than another ~300 MB docker cp.
            swap_scene(scene_dir)
        start_sim(display)
        silent = wait_for_sim(timeout_s)
        if not silent:
            if attempt > 1:
                log(f"{scene}: alive on attempt {attempt}")
            return []
        log(f"{scene}: attempt {attempt}/{max(1, attempts)} — no data on {silent}", err=True)
    return silent


def run_orchestrator(args, scene: str, report: str, append: bool) -> int:
    """One scene's questions, written into the sweep's single report.

    Every scene targets the SAME report_file. append is false only for the first scene
    that actually runs, so a re-run starts clean instead of piling onto the previous
    sweep; true afterwards, so each scene extends the file rather than replacing it.
    The orchestrator recomputes the summary on every write, so the report is complete
    and correct after each question, not just at the end.
    """
    params = [
        "-p scene_source:=sim",
        f"-p scene:={scene}",
        f"-p category:={args.category}",
        f"-p question_limit:={args.limit}",
        f"-p target_source:={args.target_source}",
        f"-p report_file:={report}",
        f"-p append:={'true' if append else 'false'}",
    ]
    if args.question_id:
        params.append(f"-p question_id:={args.question_id}")
    if args.category == 2:
        params.append(f"-p cat2_mode:={args.mode}")
    cmd = (f"source {AI_SRC}/install/setup.bash && "
           f"ros2 run smart_vlm eval_orchestrator --ros-args {' '.join(params)}")
    return sh(["docker", "exec", AI_CONTAINER, "bash", "-lc", cmd], check=False).returncode


SCENES_MOUNT = "/data/scenes"   # where <repo>/data/scenes appears inside the containers


def scene_present(scenes_dir: Path, scene: str) -> bool:
    """A scene counts only if the Unity build the sim executes is actually there.

    Same test scene_fetch uses. A folder holding map.ply but no binary would pass a
    directory check and then fail at sim launch.
    """
    return (scenes_dir / scene / "environment" / "Model.x86_64").is_file()


def fetch_scene(args, scene: str) -> bool:
    """Download a missing scene, the way bag_replay.launch fetches a missing bag.

    Runs in the AI container: gdown lives there, /data is mounted there, and writing as
    that uid keeps the files usable by both sides. Only possible when --scenes-dir IS the
    repo's data/scenes, since nothing else is visible to the container.
    """
    if args.scenes_dir.resolve() != (REPO / "data" / "scenes").resolve():
        log(f"{scene}: missing, and --scenes-dir {args.scenes_dir} is outside the "
            f"container's {SCENES_MOUNT} mount, so it cannot be fetched. Either drop the "
            f"scene there yourself, or use the default --scenes-dir to auto-download.",
            err=True)
        return False
    log(f"{scene}: not on disk — fetching (~300 MB)")
    cmd = (f"source {AI_SRC}/install/setup.bash && "
           f"ros2 run smart_vlm scene_fetch {scene} --scenes-dir {SCENES_MOUNT}")
    rc = sh(["docker", "exec", AI_CONTAINER, "bash", "-lc", cmd], check=False).returncode
    if rc != 0 or not scene_present(args.scenes_dir, scene):
        log(f"{scene}: fetch failed (scene_fetch exited {rc})", err=True)
        return False
    return True


def resolve_scenes(args) -> list[str]:
    if args.scenes and args.scenes != ["all"]:
        return args.scenes
    benchmark = REPO / "data" / "benchmark"
    folder = f"category_{args.category}"
    # Every benchmark scene for this category. Deliberately NOT filtered by what is on
    # disk: missing scenes are fetched, and filtering here would make `all` resolve to
    # nothing on a machine that has never downloaded any.
    return sorted(p.name for p in benchmark.iterdir() if (p / folder).is_dir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--scenes", nargs="*", default=["all"])
    # Beside data/bags and inside the container's /data mount, which is what makes
    # fetching possible. Override for a scene collection you already have elsewhere —
    # those still play, they just cannot be auto-fetched (see fetch_scene).
    ap.add_argument("--scenes-dir", type=Path, default=REPO / "data" / "scenes")
    ap.add_argument("--limit", type=int, default=0, help="questions per scene; 0 = all")
    ap.add_argument("--question-id", default="",
                    help="run only this question id (e.g. Q01)")
    ap.add_argument("--target-source", default="gt")
    ap.add_argument("--mode", default="hybrid", help="cat2 selection mode")
    ap.add_argument("--report", default="")
    ap.add_argument("--display", default="",
                    help="X display. Empty starts Xvfb :99 in iros2026_system "
                         "(headless EC2).")
    ap.add_argument("--sim-attempts", type=int, default=3,
                    help="how many times to start the sim for a scene before skipping it; "
                         "the first try plus two restarts by default")
    ap.add_argument("--sim-timeout", type=float, default=180.0,
                    help="seconds to wait for the sim + bridge per scene")
    args = ap.parse_args()

    # One report for the whole sweep, written by the orchestrator inside the container —
    # same file, same {"summary": ..., "results": [...]} shape a bag sweep produces. The
    # container owns the file so the uid mismatch across the boundary never arises; we
    # only ever read it here.
    container_report = args.report or f"/data/runs/cat{args.category}_sim_report.json"
    report_path = host_path(container_report)
    scenes = resolve_scenes(args)
    if not scenes:
        log("no scenes to run — check --scenes / --scenes-dir", err=True)
        return 1
    display = resolve_display(args.display)
    log(f"category {args.category}: {len(scenes)} scene(s): {scenes}  DISPLAY={display}")

    ran_any = False        # drives append: the first scene to run truncates
    failed: list[str] = []
    try:
        for i, scene in enumerate(scenes, 1):
            scene_dir = args.scenes_dir / scene
            if not scene_present(args.scenes_dir, scene) and not fetch_scene(args, scene):
                failed.append(scene)
                continue

            log(f"[{i}/{len(scenes)}] {scene}: swapping mesh and restarting the sim "
                f"(DISPLAY={display})")
            silent = bring_up_sim(scene, scene_dir, display,
                                  args.sim_timeout, args.sim_attempts)
            if silent:
                # One dead sim must not cost the rest of the sweep, exactly as one bad
                # question does not end a scene in eval_orchestrator. Skipping writes NO
                # rows, which is the point: a question that never received a frame is an
                # infrastructure failure, and scoring it 0 would understate the system.
                log(f"{scene}: no sensor data after {args.sim_attempts} attempt(s) "
                    f"(silent: {silent}) — skipping "
                    f"(see /tmp/challenge_sim.log in {SYS_CONTAINER})", err=True)
                failed.append(scene)
                stop_sim()
                continue

            log(f"{scene}: sim up — running the orchestrator")
            rc = run_orchestrator(args, scene, container_report, append=ran_any)
            ran_any = True
            stop_sim()

            # The pre-flight probe can pass and the sim still stop feeding once the question
            # is running: on the sweep that exposed this, four scenes ran their whole window
            # on odometry alone with zero camera frames. The orchestrator now notices and
            # abandons the scene WITHOUT scoring it, so one restart is worth trying.
            if rc == EXIT_SENSORS_LOST:
                log(f"{scene}: sensors died mid-question — restarting the sim and "
                    f"re-running the scene once", err=True)
                silent = bring_up_sim(scene, scene_dir, display,
                                      args.sim_timeout, args.sim_attempts)
                if silent:
                    log(f"{scene}: still no sensor data (silent: {silent}) — skipping, "
                        f"no rows written", err=True)
                    failed.append(f"{scene} (sensors)")
                    stop_sim()
                    continue
                rc = run_orchestrator(args, scene, container_report, append=ran_any)
                stop_sim()
                if rc == EXIT_SENSORS_LOST:
                    # Twice is not bad luck. Leave it out of the report entirely rather than
                    # scoring an infrastructure failure as a system failure.
                    log(f"{scene}: sensors died again — skipping, no rows written", err=True)
                    failed.append(f"{scene} (sensors)")
                    continue

            if rc != 0:
                log(f"{scene}: orchestrator exited {rc}", err=True)
                failed.append(scene)
    except KeyboardInterrupt:
        log("interrupted — writing what finished", err=True)
        stop_sim()

    # Read back what the container wrote, purely to report it. Nothing is written from
    # the host, so there is no second summary implementation to drift.
    rows = previous_results(report_path)
    if rows:
        summary = summarise(rows)
        log(f"done: {summary['correct']}/{summary['total_run']} correct "
            f"(accuracy {summary['accuracy']}, errors {summary['errors']}) "
            f"across {len(summary['per_scene'])} scene(s) -> {report_path}")
    else:
        log(f"no results in {report_path}", err=True)
    if failed:
        log(f"scenes with problems: {failed}", err=True)
    return 1 if failed and not rows else 0


if __name__ == "__main__":
    sys.exit(main())
