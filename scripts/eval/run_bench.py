#!/usr/bin/env python3
"""Eval bench orchestrator. Runs on the HOST (L4 box).

For each scene x question: swaps the scene in, relaunches the system (fresh, as
the real eval does), starts the AI module, runs qa_recorder inside the
ai_module container, collects the result JSON, tears down. Then score.py.

Usage:
  python3 run_bench.py --repo ~/sugun/CMU-VLN-Challenge-2026 \
      --scenes-dir ~/vln_scenes --out ~/vln_eval/run_$(date +%Y%m%d_%H%M) \
      [--scenes hotel_room_1 loft] [--qtypes numerical] [--smoke]

Assumptions to verify on first run (see docs/M6_eval_harness.md):
  - scene folders in --scenes-dir are named exactly like questions.json scenes
  - copying into mesh/unity + relaunching system_simulation.sh switches scene
  - DISPLAY for the system container (RViz/Unity) — set --display accordingly
"""
import argparse, json, pathlib, subprocess, sys, time

SYS, AI = "iros2026_system", "iros2026_ai_module"
MESH_SUBDIR = "autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity"


def sh(cmd, check=True, **kw):
    print(f"[bench] $ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, **kw)


def dexec(container, cmd, detach=False, display=None, check=True):
    d = "-d" if detach else ""
    env = f"-e DISPLAY={display}" if display else ""
    return sh(f"docker exec {d} {env} {container} bash -lc \"{cmd}\"", check=check)


def stop_all():
    for c, pat in [(SYS, "system_simulation|ros2|rviz|Unity|vehicleSimulator"),
                   (AI, "ros2|qa_recorder|smart_vlm")]:
        dexec(c, f"pkill -9 -f '{pat}' || true", check=False)
    time.sleep(3)


def load_questions(repo):
    qj = json.load(open(repo / "questions" / "questions.json"))
    for entry in qj:
        for qtype, qs in entry["questions"].items():
            for i, q in enumerate(qs):
                yield entry["scene"], f"{qtype[0]}{i}", qtype, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=pathlib.Path, required=True)
    ap.add_argument("--scenes-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--scenes", nargs="*", help="subset of scenes")
    ap.add_argument("--qtypes", nargs="*", help="subset of question types")
    ap.add_argument("--smoke", action="store_true", help="first question per scene only")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--display", default=":1")
    ap.add_argument("--ai-launch", default="ros2 launch smart_vlm smart_vlm.launch",
                    help="command that starts the AI module under test")
    ap.add_argument("--settle", type=int, default=25, help="s to wait after system launch")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    todo = [t for t in load_questions(args.repo)
            if (not args.scenes or t[0] in args.scenes)
            and (not args.qtypes or t[2] in args.qtypes)]
    if args.smoke:
        seen, smoke = set(), []
        for t in todo:
            if t[0] not in seen:
                seen.add(t[0]); smoke.append(t)
        todo = smoke
    print(f"[bench] {len(todo)} questions queued")

    current_scene = None
    for scene, qid, qtype, question in todo:
        res_file = args.out / f"{scene}_{qid}.json"
        if res_file.exists():
            print(f"[bench] skip existing {res_file.name}"); continue

        # 1. scene swap (host repo dir is the container's source of truth — verify!)
        if scene != current_scene:
            src = args.scenes_dir / scene
            dst = args.repo / MESH_SUBDIR
            if not src.exists():
                print(f"[bench] !! scene folder missing: {src} — skipping {scene}")
                continue
            sh(f"rm -rf '{dst}'/* && cp -r '{src}'/* '{dst}'/")
            current_scene = scene

        # 2. fresh system launch (relaunch per question, like real eval)
        stop_all()
        dexec(SYS, "/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh "
                   "> /tmp/system.log 2>&1", detach=True, display=args.display)
        time.sleep(args.settle)

        # 3. AI module under test
        dexec(AI, f"{args.ai_launch} > /tmp/ai.log 2>&1", detach=True)

        # 4. recorder (blocks until answer/timeout)
        q = question.replace('"', '\\"').replace("'", "'\\''")
        dexec(AI, "mkdir -p /tmp/eval_results", check=False)
        sh("docker cp scripts/eval/qa_recorder.py " + AI + ":/tmp/qa_recorder.py", check=False)
        rec = (f"python3 /tmp/qa_recorder.py --question '{q}' --qtype {qtype} "
               f"--scene {scene} --qid {qid} --timeout {args.timeout} "
               f"--out /tmp/eval_results/{scene}_{qid}.json")
        dexec(AI, rec, check=False)

        # 5. collect + teardown
        sh(f"docker cp {AI}:/tmp/eval_results/{scene}_{qid}.json '{res_file}'", check=False)
        stop_all()
        print(f"[bench] done {scene}/{qid}")

    print(f"[bench] all done → {args.out}\n[bench] next: python3 score.py --results {args.out} --gt gt/gt.json")


if __name__ == "__main__":
    sys.exit(main())
