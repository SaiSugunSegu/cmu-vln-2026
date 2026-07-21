# M6 — Eval harness

**Task:** Make every change measurable. Concretely: a runner that launches sim + our stack per training question (75 total: 5 × 15 scenes), captures answers + executed trajectory, scores per question type (exact match / bbox IoU vs VLA-3D GT / trajectory-vs-.ply constraint check), and reports per-module metrics.

**Status:** ⚪ not started
**Owner:**
**Depends on:** M0. Build the skeleton in W1 — every other module's "done" is defined by this one.

## Plan
| Stage | What to try |
|---|---|
| Baseline | Runner (launch, publish question, collect answer, teardown) + numerical & object-ref scorers |
| Upgrade | Trajectory scorer replicating constraint-order + penalty logic against .ply GT; per-module metrics (instance recall, duplicate rate, grounding accuracy, time-to-answer) |
| Stretch | Nightly regression on all 75 questions; ablation dashboard; parallel scene runs |

## Report format (per run)
```
scene | q_id | type | answer | gt | score | time_s | coverage% | notes
```
Plus module metrics: perception recall/precision/dupes, grounding acc (live + GT-objects mode), API cost.

## Progress checklist
- [ ] questions.json parsed; GT answers/boxes/.ply organized per scene
- [ ] Headless runner for one question end-to-end
- [ ] Numerical + object-ref scorers
- [ ] Trajectory scorer validated against provided GT examples
- [ ] Full 75-question run < some manageable wall time; results table auto-generated
- [ ] Timing instrumentation (time-to-answer distribution — tiebreaker data)

## Suggestions
- Score the DUMMY model first — establishes the floor and proves the harness before any real model exists.
- Track a single headline number (projected challenge score /45 per scene-set) so every PR answers "did the score go up?"
- Keep a fixed 10-question smoke subset for fast iteration; full 75 nightly.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created |
