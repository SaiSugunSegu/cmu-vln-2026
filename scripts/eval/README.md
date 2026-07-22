# Eval bench (M6)

Runs all 15 training scenes × 5 questions against the AI module, scores like the challenge.

```
run_bench.py     host orchestrator: scene swap → fresh system launch → AI module →
                 qa_recorder → collect JSON (relaunch per question, like real eval)
qa_recorder.py   in-container node: mimics eval node (1 Hz question), records
                 Int32 / Marker / waypoints / executed trajectory, 10-min timeout
score.py         numerical exact · object-ref 2×IoU · instruction constraint-order
                 proxy with avoid-zone penalties
gt/gt.json       curated ground truth (fill from answer PDFs + VLA-3D + .ply)
```

## Quick start
```bash
# smoke run (1 question per scene, dummy model):
python3 scripts/eval/run_bench.py --repo . --scenes-dir ~/vln_scenes \
    --out ~/vln_eval/smoke --smoke
python3 scripts/eval/score.py --results ~/vln_eval/smoke --gt scripts/eval/gt/gt.json
```

## First-run verification list (v0 — untested assumptions)
1. **Scene swap mechanism**: does copying into `…/vehicle_simulator/mesh/unity/` on the HOST
   affect the container (volume mount?), or does it need `docker cp` / image rebuild? Fix
   `run_bench.py` step 1 accordingly.
2. **System readiness signal**: recorder uses first `/registered_scan`. Confirm state
   estimation is actually converged by then; else add settle margin.
3. **Done-detection windows** (20 s marker settle, 30 s waypoint idle): tune against dummy model.
4. **Marker → GT overlap**: we proxy "degree of overlap" as 3D IoU × 2. Ask organizers
   (GitHub issue) what the real function is.
5. **Instruction scorer semantics**: validate our proxy against the provided `.ply`
   trajectories + answer images; refine radii (default 1.5 m) and penalty (−1.5/violation).

## GT curation (the real work — ~1–2 days)
Per scene: numerical integer from `questions/<scene>.pdf`; object-ref target box from the
VLA-3D object list (match label + answer image); instruction constraints (ordered pass-near
centers, avoid polygons, goal) from the statement + `.ply`. Fill `gt/gt.json`; `score.py`
flags every question still missing GT.

## Outputs
Per-question JSON (answers + trajectory + timing) in `--out`; score table + CSV; keep runs
under `~/vln_eval/<date>_<tag>` and log the headline number in `docs/M6_eval_harness.md`.
