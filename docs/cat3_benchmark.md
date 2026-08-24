# Category-3 (instruction-following) benchmark

Ground truth for the 30 official instruction-following questions — 2 per scene × 15 scenes —
recovered from the organizers' demonstration trajectories.

QA data lives under [`data/benchmark/`](../data/benchmark/).

```
data/benchmark/
  <scene>/category_3/<scene>_category3_qa.json    # 2 questions per scene, 30 total
```

Category 3 is worth **6 of the 9 points per scene** ([README](../README.md), "Question Types
and Initial Scoring") — more than categories 1 and 2 combined. [M5](M5_instruction_planner.md)
names the risk: *"Build our own scorer FIRST — iterating blind on 6-point questions is the
biggest schedule risk in the project."* This file is what that scorer scores against.

## Why the ground truth had to be recovered

The organizers publish the questions and nothing else. Every `questions/<scene>/questions.pdf`
says only:

```
Category 3: Instruction Following
Question 4: Go to the cup near the TV remote and avoid the path near the cabinet
Response: Trajectory (series of waypoints)
```

No radius, no tolerance, no definition of "near" or "between", and no mention of the `.ply`
files. The README says the score depends on "whether it follows the path constraints in the
command and in the correct order", with penalties for wrong order, unmet constraints, and
entering forbidden areas — but `challenge_evaluation_node` is not public.

What *is* published is `questions/<scene>/trajectory_q4.ply` and `trajectory_q5.ply`: one
dense recording of a correct run per question, ~1 cm between vertices. **Those recordings are
the only empirical evidence of what this challenge means by "near".** The benchmark is a
reading of them.

## The two findings that make this exact

**1. The trajectory frame and the IRef-VLA annotation frame are the same frame.** No offset,
no axis swap, no scale. Every demo starts within 1.5 cm of (0, 0) — the robot spawn, since
`vehicleX`/`vehicleY` default to 0 — and every vertex sits at `z = 0.75`, the `vehicleHeight`
in `vehicleSimulator.cpp:49`. Confirmed semantically, not just by bounding boxes:

| scene | q | the instruction's final stop | trajectory endpoint → object |
|---|---|---|---|
| loft | q4 | "the cup near the TV remote" | `108 cup` **0.52 m**, `18 tv remote` **0.58 m** |
| loft | q5 | "the sphere decoration on the cabinet" | `56 sphere decoration` **0.96 m** |
| office_1 | q4 | "the water cooler near the window" | `23 water cooler` **0.78 m** |
| livingroom_2 | q5 | "the soccer ball near the couch" | `63 soccer ball` **0.58 m** |
| studio | q4 | "the guitar near the couch" | `48 guitar` **0.72 m** |

All 30 endpoints land 0.25–1.7 m from the object their sentence names as the stop.

**2. Position along the path is the constraint order** — the thing actually being scored.
loft q5 → `12 fireplace` at vertex 117, `69 stairs` at 330, `56 sphere decoration` at 758 of
771, which is exactly "go near the fireplace, pass by the stairs, then stop at the sphere
decoration".

Note `z = 0.75` is the sensor plane, not the floor (floor is `z ≈ 0`). Compare `(x, y)` only.

## How a question is built

`scripts/bench/generate_category3_qa.py`, per question:

1. **Split the instruction at its verbs** into ordered clauses. The verb decides the sign —
   "take the path between the two columns" and "avoiding the path between the chair and the
   folding screen" are the same geometry with opposite meaning.

   | clause | kind |
   |---|---|
   | `go to`, `go near`, `first, go …` | `pass_near` |
   | `take the path near X`, `pass by X` | `pass_near` |
   | `take the path between A and B`, `go between A and B` | `pass_between` |
   | `stop at`, `stop by`, `and finally, to …` | `goal` |
   | `avoid(ing) the path near/between …` | `avoid_near` / `avoid_between` |

   A trailing `to` inside a clause opens a second constraint — "take the path between the TV
   and the bed **to** the picture closest to the TV" is a pass and a goal — but only when it
   is not owned by a comparative, so "closest to the TV" stays one constraint.

2. **Ground each clause** with `utils.text_solver.solve`, the same solver the category-2
   benchmark is generated and audited with, so "closest" means one thing across both.

3. **Cross-examine the grounding against the recording.** Each constraint is looked for in
   the part of the run still ahead of the previous one — the same cursor `score_instruction`
   advances — so an incidental early pass cannot be credited. A goal is measured from the
   *final* pose, because that is all a goal is scored on.

4. **Let the demo settle what the text cannot.** Where the solver is indecisive (two
   cabinets in two rooms both match "the cabinet"), the candidate the run actually approaches
   wins, but only by ≥ 0.30 m over its runner-up — the same margin category 2 requires.
   Co-located candidates that are one place to stand (four vases on a cabinet, a window
   annotated as two panes) become one waypoint.

5. **Anything still unsettled is not written as ground truth.** It goes to
   `scripts/bench/category3_candidates/<scene>.json` with its alternatives ranked by what the
   demo did, for a human to decide in `scripts/bench/category3_overrides.json`.

**25 of 30 questions ground with no human input; 4 more needed one decision each; 1 is a
recorded deviation** (below).

### Reference objects: the nouns you have to find but never drive to

"Go near the stool **under the picture** and stop at the small table **farthest from the
columns**" names four things and sends the robot to two. The picture and the columns are
not destinations — they are how you tell *which* stool and *which* table. A system that
never detects them cannot ground the two that are scored, so the benchmark carries their
boxes too, tagged `role: "reference"` against the destinations' `role: "constraint"`.

They are documentation of the grounding, never geometry: no reference object moves a
`center`, a `radius` or a `polygon`.

They are resolved from the *already grounded* target rather than from the solver's
internals, so a clause pinned in `category3_overrides.json` gets its references the same
way a solved one does. Per noun phrase in the clause, in order:

| stage | rule |
|---|---|
| colour | drop colour words before matching — `match_class("black chair")` finds nothing, which is how livingroom_1's "closest to the black chair" loses its reference |
| class | `match_class(…, strict=True)` inside the target's region, falling back scene-wide when the region holds nothing by that name — hotel_room_2's "the picture closest to the door" needs the fallback, its door frame being annotated in a different region |
| relation | `on` / `supports` / `above` / `below` / `in` / `near`: keep the candidates the boxes actually put there. This is what picks `picture#3` out of arabic_room's two pictures |
| between | keep the pair that brackets the target most squarely, by `between_holds`' lateral offset — livingroom_1's vase satisfies three different tv/door pairs at once |
| comparatives | `closest` / `farthest`: no filter. The candidate set *is* the comparison |
| plural | a phrase naming a set keeps all of it: "the columns" means both |
| singular | still undecided → the nearest by box gap, which is the object the solver ranked on too (`set_dist` is a `min` over the anchor set, so even "farthest from" is decided by the closest member) |

Each constraint's `anchor_why` records which rule chose what. Two are worth naming because
the phrase outruns the annotation:

- **home_building_1 Q04**, "the dining table near **the big picture**" — the region holds
  four pictures and IRef records no size adjective, so the nearest, `photo#162`, is the one
  kept.
- **livingroom_1 Q05**, "the cabinet **with a picture above it**" — the parser reads "with"
  as `supports`, and no picture rests on that cabinet, so no candidate is filtered by the
  relation and the nearest of the seven, `picture#88`, is kept.

Both are recorded in the file rather than smoothed away. Neither affects a score.

### "Between" means what VLA-3D means by it

`pass_between` and `avoid_between` regions use `utils.geometry`'s constants
(`BETWEEN_SPAN`, `BETWEEN_LATERAL`, `BETWEEN_LATERAL_MAX`) — the middle stretch of the
segment joining the two centres, widened proportionally and capped at a metre. VLA-3D's
`between` heuristic is what generated these questions in the first place, so the avoid zone
means what the question meant.

The obvious alternative — the convex hull of the two footprints — is wrong: it spans the
objects themselves and everything outboard of them. In livingroom_2 it swallowed half the
room and flagged a run that passed nowhere near the TV as a violation.

`avoid_near X` is X's footprint grown outward by ~1 m, since X's own footprint is space the
robot could not occupy anyway.

### Radii

`radius` defaults to 1.5 m, `score.py`'s value. Eight constraints across five scenes carry a
wider one, always with a `radius_source` saying why: a wall-mounted target cannot be
approached. `hotel_room_2` q5 views its picture from 2.64 m because there is a bed in the
way, so its radius is 2.9 m. Widening only ever happens where the demo *does* satisfy the
constraint at that distance — a constraint the run never meets keeps 1.5 m and fails loudly
rather than being papered over.

## File schema

Top-level keys mirror categories 1 and 2 (`scene`, `category`, `category_name`, `description`,
`notes`, `difficulty_counts`, `target_object_coverage`, `questions`). Per question:

| field | meaning |
|---|---|
| `target_objects` | every noun the instruction names, in its order — destinations *and* references. This is the SAM-3 prompt list |
| `gt.pass_near[]` | **ordered**, excludes the goal. `center` + `radius` are what a scorer needs; `object_ids`, `traj_index`, `traj_frac`, `min_dist_m`, `why` are the evidence |
| `gt.*.anchor_ids` | the clause's reference objects, with `anchor_why` saying how each was chosen |
| `gt.avoid[]` | `polygon` in the map frame, tested against the whole path |
| `gt.goal` | same shape; scored on the final pose |
| `objects` | every referenced IRef-VLA box (`center`, `size`, `yaw`, `aabb`, `bbox_corners`), in instruction order, each tagged `role: constraint` or `role: reference`, so the file is self-contained |
| `reference_trajectory` | the `.ply` path, vertex count, length, endpoints, and the run resampled to ~0.25 m |
| `evidence`, `solver_trace` | how the grounding was reached |
| `verified` | false when a check failed; `review_note` then says what |

The `gt` block is shaped so `scripts/eval/score.py::score_instruction` consumes it unchanged
— it reads `center`/`radius`/`polygon` and ignores the diagnostic keys beside them.

`traj_index` / `traj_frac` / `min_dist_m` record where the *organizers'* run satisfies each
constraint. They are evidence for the grounding, not something a system under test has to
reproduce.

30 questions: 14 easy, 13 medium, 3 hard (hard = carries an avoid zone). 33 `pass_near`,
9 `pass_between`, 30 goals, 3 avoid zones. 154 boxes in all — 94 constraint, 60 reference.
Difficulty counts constraints only; a reference object never makes a question harder to
score, only harder to ground.

## Verifying

```bash
just cat3-verify              # or: python3 scripts/eval/verify_category3.py [scene ...] -v
```

Runs on the **host**, not in a container — it reads `../IRef-VLA`, which nothing mounts (same
reason as `just map3d-priors`). No GPU.

Seven checks per question, but one is the gate:

> **Replay each demo through `score.py::score_instruction` and require 6.0/6.**

That is the same scorer the eval harness runs on our own executed trajectories, fed the
reference run in the same `(stamp, x, y)` shape `qa_recorder.py` produces. A passing entry
means the organizers' own run, scored exactly as we will score ourselves, earns full marks.
A too-tight radius, a wrong order, or an avoid zone drawn over the demo's own path all fail
here. The other six checks confirm the boxes still match the IRef-VLA metadata, the order is
monotonic, no avoid zone contains the demo, `reference_trajectory` still describes the
`.ply` on disk, every carried box is cited by a constraint with the matching `role` (and
every cited object is carried), and `target_objects` is exactly the nouns of the `objects`
block — the prompt list cannot drift from the boxes it is meant to cover.

Current state: **29/30 verified at 6.0/6, 1 known deviation.**

This closes open question #5 in [`scripts/eval/README.md`](../scripts/eval/README.md)
("validate our instruction proxy against the provided `.ply` trajectories; refine radii and
penalty") — the proxy now has something to be right about.

## Rebuilding

```bash
just cat3-build                                            # --dry-run: report, write nothing
just cat3-build "--dry-run --scene loft -v"
just cat3-build --write                                    # regenerate all 15 files
python3 scripts/bench/generate_category3_qa.py --plot loft  # top-down PNG for eyeballing
```

Regeneration is deterministic: hand decisions live in `category3_overrides.json`, so
`--write` followed by `git diff --exit-code data/benchmark` is clean.

`--plot <scene>` writes `scripts/bench/category3_candidates/<scene>.png` — the demo path with
anchor boxes, constraint radii and avoid polygons overlaid — for comparison against that
scene's `questions.pdf` answer screenshot.

## The curated decisions

All four are recorded in `scripts/bench/category3_overrides.json` with their evidence.

- **chinese_room Q05** — the scene has no object labelled "folding screen"; the carved
  partition is annotated `partition wall` #36 (4.65 m long, 7 cm thick, versus 7–9 m for the
  real walls). Which chair is fixed by elimination: the demo passes through the between-band
  for chairs #29, #61, #81 and #86, so none can be the one it was told to avoid. Of the two
  remaining, #84 is nearer the screen. A judgement call — the sentence does not distinguish
  #84 from #60, and both leave the demo compliant.
- **home_building_1 Q05** — "the picture" is photo #76, the only picture whose band with
  dining table #266 the run traverses (150 vertices inside, gap centre 0.66 m off the path).
- **loft Q04** — two cabinets match and the demo avoids both. #17 is the TV cabinet, in the
  same region as the cup the instruction targets; #35 is in a room this run never enters,
  which would make the constraint vacuous. Loft's own Q03 uses "the cabinet" the same way.
- **office_2 Q05** — folders #69 and #70 sit 7 cm apart on cabinet #56, so they are one
  waypoint either way. #56 is not the cabinet geometrically closest to the whiteboard (#160
  is), but #160 holds only a phone and a box; among cabinets carrying a folder at all, #56 is
  closest.

### The one deviation: home_building_1 Q05

The recording and the sentence disagree about **order**, and the recording is what it is.

The instruction is "nightstand → between the dining table and the picture → trash can". The
33 m run through this multi-room building passes between dining table #266 and photo #76 at
1% of the way, reaches nightstand #427 at 43%, and ends at trash can #103. After the
nightstand it never goes between any table and any picture again — the nearest such gap
centre in the remaining two thirds of the run is 3.13 m off the path.

So this demo scores **5.00/6** against its own instruction (`score_instruction` gives the
middle constraint half credit as out-of-order). The entry keeps the constraint grounded to
what the run demonstrably does, marked `verified: false` with a `review_note`, rather than
inventing a leg the recording does not contain. `verify_category3.py` reports it as a known
deviation and excludes it from the pass count.

## Scope

This is benchmark data plus its generator and verifier — the thing a category-3 run is scored
against, not the thing that answers it.

The runtime is now wired to it: `eval_orchestrator.py` runs `category:=3`, recording
`/state_estimation` for the whole question and grading the driven path with the same
`score.py::score_instruction` this file's gate uses, and `just eval-cat3-sim` sweeps it against
the live sim. The system under test is `instruction_reasoner`, which plans a route with one VLM
call over the robot's own map — [cat3_vlm_contract.md](cat3_vlm_contract.md) is that contract.

Nothing in that runtime path reads this directory, and nothing in it shares the clause parser
that generated these files. That separation is deliberate: a system graded on ground truth its
own parser produced would be grading its reading of the sentence against itself.

## Related

- [cat3_vlm_contract.md](cat3_vlm_contract.md) — what the runtime asks the VLM, and what it does with the answer
- [`scripts/bench/generate_category3_qa.py`](../scripts/bench/generate_category3_qa.py) — generator
- [`scripts/eval/verify_category3.py`](../scripts/eval/verify_category3.py) — the audit
- [`scripts/utils/geometry.py`](../scripts/utils/geometry.py), [`text_solver.py`](../scripts/utils/text_solver.py) — shared grounding, also used by category 2
- [M5 — Instruction-following planner](M5_instruction_planner.md) · [M6 — Eval harness](M6_eval_harness.md)
- [Category-1 bag benchmark](cat1_bag_benchmark.md)
