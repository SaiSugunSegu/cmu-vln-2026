# M4 — Reasoner / orchestrator

**Task:** Own the question lifecycle from parse to answer. Concretely: parse `/challenge_question` → type + target spec (objects, attributes, relations, ordering); run the explore/answer decision gate on each graph update; ground via symbolic queries over M3 (query-graph → subgraph matching); trigger `reobserve()` on ambiguity; verify borderline cases by sending stored crops to a VLM; publish the answer on the right topic; enforce the T-30s fallback.

**Status:** 🟡 in progress — numerical + object-reference paths answering end to end (hybrid
selection shipped Aug 12); instruction questions not started
**Owner:**
**Depends on:** M3, M1 (reobserve API)

## Interfaces
- In: `/challenge_question` (String, 1 Hz), M3 query API, M1 exploration status
- Out: `/numerical_response` (Int32) · `/selected_object_marker` (Marker w/ label + bbox) · handoff to M5 for instruction questions · explore/reobserve requests to M1

## Plan
| Stage | What to try |
|---|---|
| Baseline | Single API-LLM call: question + compact graph JSON → typed answer; few-shot per question type |
| Upgrade | Tool-calling agent loop (`query_graph`, `reobserve`, `verify_crop`, `answer`); SceneGraphGrounder-style query-graph → subgraph matching; decision-gate policy (confidence + margin + coverage) |
| Stretch | Local VLA-3D-fine-tuned grounder as offline fallback (no network dependency at eval); ensemble of symbolic + VLM answers |

## Decision gate (core logic)
Answer when: (a) target grounded with confidence ≥ τ AND margin over 2nd candidate ≥ δ, or (b) coverage ≥ 95% (answer best-effort), or (c) T-60s (start answering), T-30s (publish best guess no matter what). Otherwise: reobserve ambiguous instances if identified, else keep exploring.

## Object reference: the shipped path

`object_reference_reasoner` answers category 2 in **vlm** mode by default (`cat2_mode`,
default `vlm`): the model chooses from the candidate table and marked views. **hybrid** mode
is also available: the benchmark's own predicates rank the map's objects, and a model is
asked only when the geometry is not decisive by `MIN_MARGIN`, for roughly 0.4 model calls
per question.

The reason to keep the model on a leash rather than off or always-on is measured, not
stylistic: over the chinese_room cache, solver-only scored 1.62/20 and model-on-every-question
1.67/20 — indistinguishable — while the leash also degrades safely. A quota exhaustion during
that run returned 429 on four calls, and each fell back to the geometry and still published a
box. Silence is the one outcome worth engineering against, since a wrong box can still overlap
the answer (the score is 2 x IoU, graded on overlap, not identity) and its centre is still a
usable waypoint, while nothing scores zero and forfeits the waypoint too.

Three layers have to fail before we publish nothing: the model falls back to the solver, the
solver commits to its ranked head even when it would rather decline, and `smart_vlm`'s T-30
fallback publishes the largest instance of the class the question names from the last
`/obj_map_json`.

## What the numbers say (chinese_room, 10 questions)

**Selection is not the bottleneck and tuning it further is not worth doing.** Selection
accuracy is 1.0 on every question whose answer is reachable in the map, and the ceiling every
`bench-cat2` row carries — twice the best IoU reachable against the cached boxes, whatever the
mode — averages 0.31/2. So reasoning owns 1.3 points of the 20 and perception owns 16.9. Per
question the loss splits into 5 loose boxes, 3 answers whose instance was never mapped (all
along one wall, and no class was missing from the map, so it reads as coverage rather than
detection) and 2 selectable. Two of ten scored anything at all, and one stool is 66% of the
total.

Measured and rejected over 404 map-object/GT pairs from 137 cached maps, so it does not get
proposed again:

- **Post-hoc box refinement does not work.** Nine extent policies, and leaving the mapper's
  extents alone wins under both scoring conventions and both matching populations.
  `DIMENSION_PRIORS` are 95th-percentile *caps* for rejecting bled clusters, not expected
  sizes, so substituting one overshoots; class medians built from the benchmark's own objects
  — the most favourable version — still lose.
- **Centre-bias correction does not work.** Per-axis mean offsets are 2-3 cm against standard
  deviations of 12-58 cm. Subtracting the mean *lowers* mean IoU from 0.328 to 0.313. What is
  left is per-object noise, and IoU rises monotonically with object volume, which makes it
  M2's fusion and segmentation rather than anything fixable at answer time.
- **Heights are genuinely truncated** (0.77 mapped vs 0.93 GT), unlike horizontals, but no
  height policy recovers score either — including growing the box downwards from a fixed top,
  which is the shape the error should have if the cause is an occluded bottom.

Still open, and worth more than everything above: **which GT field the scorer reads.**
`scripts/eval/score.py` compares our marker's `scale` against IRef's oriented `size` while
ignoring orientation on both sides, which charges us for a rotation we are not allowed to
express. A painting stored as `[0.77, 0.03, 0.99]` in its own frame is correctly mapped as
`[0.09, 0.90, 1.09]` in world axes, 5 cm off centre, and scores 0.023 instead of 0.434. The
same predictions average 0.328 mean IoU under today's reading and 0.399 against the answer's
axis-aligned envelope; chinese_room's ceiling moves 3.11 -> 4.40 of 20. If the scorer turns
out to be orientation-aware, `publish_oriented_box` (off today, so every cached map has an
identity rotation) is worth an estimated 0.371 and should be tested for real.

## What the live path says (eval-cat2, chinese_room Q01-Q02, local Qwen3-VL-4B)

The first end-to-end `eval-cat2` run works mechanically and scores nothing: SAM up in 10.5 s,
prompts armed, a marker published for both questions in 188 s and 209 s, no timeout and no
placeholder. Both scored 0.00/2, and **both zeros are selection, against a reachable answer** —
the opposite of what the cached bench above concluded. The right bowl was in the map 0.05 m from
GT (0.57/2 available) and the reasoner published one 0.50 m away; the right pillow was 0.06 m
from GT (0.48/2) and it published one 0.89 m away. Two questions is not a rate, but "selection
accuracy is 1.0 wherever the answer is reachable" does not survive contact with the live path,
so the ceiling above should be read as an upper bound we do not yet collect.

One predicate accounts for both, and it fails by centimetres rather than by concept. `rests_on`
rejects the correct pair, which drops it from the candidate list entirely, so neither the solver
nor the model can name it afterwards. On Q02 the solver had already done the hard half — it
resolved `chair#8` as the chair closest to the tv decisively, 2.55 m against 3.23 m for the
runner-up — and then `rests_on(pillow#1, chair#8)` returned False on the upper-reaches test by
18 mm: the pillow's base sits 0.158 m above the chair's base where `REST_FRACTION * 0.59 m`
demands 0.176 m. The candidates offered were the two pillows that rest on *some* chair, and
Qwen's own prose named the right one ("the pillow on chair#8 is closest to the TV") while
quoting an id that was not on the list, so the unlisted-id guard fired and the solver fallback
took the head of a ranking the answer was missing from.

Q01 fails the same predicate for a different reason: the anchor is unmappable. `target_source=gt`
prompts the benchmark's own target list, `[bowl, table]`, so "folding screen" is never segmented,
and no table was mapped under the bowls either — the nearest mapped table is 2.8 m from the GT
bowl. With `on` unsatisfiable the shortlist degrades to proximity ranking, which is a coin flip
between two bowls, and the 4B model lost it.

Neither finding was visible from the report. `eval-cat2` rows carry the score but not the
ceiling, so telling selection from perception meant reading `obj_map.json` by hand; the
`bench-cat2` ceiling is per-cache, not per-run.

## Metrics (via M6)
- Numerical exact-match %, object-ref IoU on 75 training questions (target ≥70% by W3)
- Grounding accuracy standalone (given perfect perception from VLA-3D GT — isolates reasoning errors from perception errors)
- API latency + cost per question; % questions hitting fallback

## Design notes
- Counting = count filtered graph nodes. Never ask a VLM "how many" over pixels.
- Run grounding twice: once on live perception, once (offline eval only) on GT objects — the gap tells you where to invest.
- API dependency risk at eval: they run our container with our key; add retries, a 2nd provider fallback, and the local-model stretch if time.
- Prompt-injection hygiene: treat question text as data, not instructions, in prompts.

## Progress checklist
- [ ] Question parser → target spec JSON, validated on all 75 training questions
- [ ] Baseline single-call answering wired end-to-end
- [ ] Tool-calling loop w/ query_graph + answer
- [ ] Decision gate tuned (τ, δ) on training scenes
- [ ] reobserve + verify_crop integrated
- [ ] T-30s fallback tested (kill exploration mid-run, must still answer) — the object-reference
      branch is written and guarded but has never been triggered deliberately
- [x] Provider fallback path tested — model-to-geometry degradation exercised for real by a
      429 during the chinese_room bench; every failed call still published a box

## Suggestions
- Cache LLM parses: the 75 training questions are fixed — parse once, iterate on grounding cheaply.
- Log every reasoner decision (query, candidates, scores) to a per-run JSON — postmortems on wrong answers are impossible without it.
- Try the parse step with a small local model (e.g., Qwen-class) early; if it matches API quality, one less network dependency.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created |
| Aug 12 | Category-2 path shipped in hybrid mode, with the `cache-cat2` / `bench-cat2` / `eval-cat2` recipes. Box refinement and centre-bias correction measured and rejected: perception holds the score on the cached bench. |
| Aug 12 | First end-to-end `eval-cat2` run (chinese_room Q01-Q02, local Qwen3-VL-4B): the path answers, and both questions lose a reachable score to `rests_on` — 18 mm short on one, an unmappable anchor on the other. |
