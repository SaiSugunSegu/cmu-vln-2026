# Category-3: what the VLM is asked, and what is done with the answer

Instruction following is answered by **driving**, not by publishing a message: the challenge
scores "the actual trajectory followed by the robot" ([README](../README.md), *Question Types
and Initial Scoring*). This file is the contract between the model and the robot — what it is
shown, where those inputs come from, and exactly how its reply becomes wheel motion.

The live copy of the prompt is
[`captioner/prompts/instruction_planning.py`](../ai_module/src/captioner/captioner/prompts/instruction_planning.py).
Keep the two in step.

## Two calls

```
 t=0   question on /challenge_question
        |
        +-- CALL 1  EXTRACT   text only, ~1 s
        |     in : the command sentence
        |     out: TargetList  ->  /sam3/set_prompts
        |
        |   SAM arms -> TARE explores -> map_node writes obj_map.json + best-view frames
        |
 T-210  /pipeline/explore_done
        |
        +-- CALL 2  ROUTE     the reasoning call
        |     in : the command
        |        + the whole map as a table (id, label, centre, size)
        |        + 2-3 best-view frames, every mapped object outlined and tagged with its map id
        |        + the robot's current position
        |     out: RoutePlan - ordered waypoints, each with x, y, object_ids, role, why
        |
        +-- validate -> drive -> close on the goal -> hold -> idle
```

Call 1 is the existing extraction call, unchanged in shape
([`object_extraction.py`](../ai_module/src/captioner/captioner/prompts/object_extraction.py),
scored on its own by `just eval-target-extract --category 3`). It gates everything: a class it
fails to name is a class SAM never detects, and an object never detected cannot be reasoned
about at all.

## Where each input comes from

| input | produced by | notes |
|---|---|---|
| the command | `/challenge_question`, published at 1 Hz by the evaluation node | one sentence, once per system launch |
| the object table | `obj_map.json`, written by `sam_mapper/map_node.py` beside the run's crops | id, label, 3D box centre and size in the map frame. Labels come from SAM and are sometimes wrong; boxes are usually right |
| the images | the best-view frames `sam_node` saved, picked by `vqa.yaml`'s `view_source` | `full_silhouette` sends the whole 360 panorama with every mapped object outlined and tagged `[map id] label`, drawn by sam_node's own finalize pass so the tags and the table agree. Nothing is re-drawn by the reasoner |
| the robot position | `/state_estimation` | same map frame as the boxes |

Nothing else is available, and nothing else is allowed: these all derive from the six topics
the README permits at test time.

## Why there is no solver

Categories 1 and 2 rank candidates and evaluate spatial predicates in code before the model
sees anything. Category 3 deliberately does not:

- **The maps are small.** A scene runs to a couple of dozen usable rows, so the whole map fits
  in one prompt. Ranking exists to keep a long table readable; here it would only hide objects.
- **Pruning removes the destinations.** The category-2 shortlist drops room-scale structure,
  and columns, stairs, windows and door frames are places instructions send the robot to in a
  third of the corpus.
- **The failures a solver cannot fix are the ones a picture can.** In a recorded `arabic_room`
  run the object the command called a *stool* was labelled `table` by the mapper. No
  label-and-geometry solver grounds that phrase; a model looking at the tagged crop does.

The cost is that a flash model doing arithmetic over coordinates will occasionally flip a
comparative where code never would. That is why every waypoint carries a `why`, and why the
whole exchange is written to `instruction_plan.json`.

## Which input is the authority

Three tiers, weighed in this order and reasoned over rather than taken blindly:

| rank | source | why it sits here |
|---|---|---|
| 1 | **the image itself** | the plainest evidence of what is really in the room |
| 2 | **the silhouette outline and its `label [id]` tag** | drawn by SAM 3 and usually right; the label and id stay consistent across views *and* with the object table, so agreement across two views is strong |
| 3 | **the object table's row** | built from those same detections, so where it disagrees with what is plainly visible, the row is what is wrong |

The caption really is `label [id]` — `best_view.py` builds it as `f"{label} [{map_id}]"`, where
`map_id` is a key of `obj_map.json`. That join is the whole point: it is the only handle tying
a pixel region to a row the model can cite in `object_ids`.

This ordering governs **identity** — *which* object a phrase names. Coordinates are a separate
matter and rules 7 and 8 of the prompt handle them: positions are copied from the table, and a
waypoint that cites ids must sit on them, because `parse_route` replaces any coordinate more
than `SNAP_M` (1 m) from its cited ids' centre.

The three frames are **one room from several viewpoints**, each a 1920×640 equirectangular
panorama: 1920 px over 360° and 640 px over 120° are both 5.33 px per degree, so the horizontal
axis is bearing and the two side edges are adjacent. The prompt says so, because an object can
straddle the seam and appear at both edges of one frame, and because something occluded in one
view is often plain in another. Every outline is captioned `[map id] label` from
`obj_map.json`, which is the only handle tying pixels to a row the model can cite.

## When the map is missing the target

About **one named object in seven** is never detected on a live run, and it is almost always a
small thing resting on a large one — tray, kettle, cup, remote, figurine. The prompt has the
model place it from the mapped object it sits on rather than dropping the waypoint, because the
position survives the substitution: across the corpus **40 of 51 ground-truth constraint centres
sit within 1.5 m of some mapped object, median gap 0.30 m**. A dropped constraint scores zero;
an anchored one is inside the scoring radius about 78% of the time.

The waypoint carries the *anchor's* id in `object_ids`, which is also what keeps it inside
`parse_route`'s snap check. There is deliberately no estimated bounding box: `RouteWaypoint`
has no field for one and nothing downstream would read it — the route needs `x, y`.

When the model can neither see the object nor tie it to a mapped one it leaves that waypoint
out, except at the goal, which is never dropped.

## What the reply must satisfy

`RoutePlan` (see
[`vlm_backends/schemas.py`](../ai_module/src/captioner/captioner/vlm_backends/schemas.py)) is a
`reason` plus an ordered `waypoints` list. Each waypoint is `role` (`pass` | `goal`), `x`, `y`,
the `object_ids` it stands at, and `why`.

`cat3_utils.parse_route` owns only the bookkeeping — the model owns every decision:

| check | why |
|---|---|
| unknown `role` becomes `pass` | a mislabelled waypoint is still a place worth visiting |
| cited ids absent from the map are dropped | the model may cite a row that was filtered as noise |
| a coordinate more than `SNAP_M` (1 m) from its cited objects is replaced by their centre | catches a fabricated or transposed number without second-guessing a legitimate midpoint |
| exactly one `goal`, moved to the end | the goal is the only constraint scored on where the robot *ends*; a route finishing elsewhere throws it away |
| every waypoint carries the same `reach_m` = `settle_radius_m` | one rule and one distance; see *How the reply becomes motion* |

Rows whose box is under 6 cm on every axis never reach the model at all (`usable_objects`):
`map_node` emits such a stub for a track that never accumulated geometry, and a live map
carries one sitting at exactly the origin — the robot's own spawn point.

## How the reply becomes motion

**The route is exactly the waypoints the model returned** — nothing interpolated, offset or
inserted. An earlier version split long legs into 3.5 m hops; the first sim run then wedged for
30 s on a hop it had invented, at literally zero displacement, and freed itself the moment it
was re-aimed at the real waypoint 4.6 m away. Every waypoint the *model* chose was reached
without incident.

What *is* adjusted is where each waypoint is published, by `cat3_utils.snap_to_traversable`.
The model reasons about object **centres**, so a correct waypoint routinely lands inside the
furniture it names — livingroom_1's "between the sofa and the round tables" is a midpoint that,
against ground truth's own boxes, sits inside the sofa *and* inside the table. Something has to
move it onto floor.

**One rule: the closest point of the traversable area.** Nothing is traded against that distance.

That area is `smart_vlm/traversable_area.py` — a grid of the floor the robot has actually seen,
built from `/terrain_map_ext` split on the base autonomy's own `obstacleHeightThre` (0.05 m), fed
at 1 Hz from the moment the stack launches. It has to be accumulated and it has to start early:
terrain analysis runs `decayTime = 4.0` with `noDecayDis = 0.0`, so any single frame holds only
the last few seconds of line of sight, and ground near a waypoint 6 m away is simply not in it.
Building through exploration means that by the time a route exists, the floor the robot drove past
is already there. It also has to be ours to build — the challenge **withdrew the traversable area
this year**, so terrain analysis is the only reading of the floor allowed at test time.

Three cell states, and the third one matters: a cell nobody has looked at is `UNKNOWN`, not free.
Aiming at ground that was never observed is how a waypoint ends up inside furniture the map has no
reading for.

### Why a grid, and what it cost before

| | per terrain message |
|---|---|
| accumulate points, `vstack` + `np.unique` | 37 ms at 50k accumulated, 101 ms at 165k, 241 ms at 400k — **growing all run** |
| scatter into a fixed grid | 0.11 ms at 10k points, 0.86 ms at 60k — **constant**, 0.16 MB for a 40 m square |

The old cost was O(everything seen so far) on every message, which is why terrain used to be
throttled to one fold per 0.25 m of travel. The grid is O(points in *this* message), so it is just
read at 1 Hz. The query changed the same way: one `distance_transform_edt` answers "which free cell
is nearest" for every cell at once (9.6 ms), and each snap after it is an array index — **11 µs,
against 104 ms** for the pairwise scan it replaced. Re-snapping every metre got cheaper, not dearer.

The cell is 0.10 m, matching terrain analysis' own `scanVoxelSize`. Finer marks a lattice with
unmarked holes between the source's points rather than a filled region: no more accurate, four
times the transform.

### What was removed, and why

The previous snap minimised

```
||p - waypoint|| + 0.10 * ||p - vehicle||        over floor eroded by 0.75 m
```

Both terms pushed the aim point away from the waypoint deliberately. The 0.75 m erosion was the
converter's own `obstacleDisThre`, copied so that a point we chose was one it would accept and pass
through untouched (`waypointConverter.cpp:217-222`); the vehicle term broke ties toward the robot's
side. Measured over `livingroom_1` Q04+Q05 the two together moved **every** waypoint 0.94–1.41 m,
and since the robot tracks what we publish to within 0.3 m, that displacement *was* the error.

Both are gone. The vehicle term is deleted outright — it was a bias, never a reachability guard,
and it contradicts aiming at the closest point. `snap_clearance_m` survives as a config knob
defaulting to **0**, so the erosion can be bought back in whatever amount is wanted without a code
change.

**The consequence, stated plainly:** at zero clearance the published point can sit closer to
furniture than the converter's own candidate filter allows, so it may re-aim what we publish rather
than pass it through. That is an accepted trade, not an oversight. `snap_m` drops sharply;
`final_m` may improve by less, because where the robot stops is then the base stack's decision.
`final_target_m` is the field that shows which happened.

`max_snap_m` (3 m) is the honesty limit and stays: past it the nearest reading is not the edge of
the object the model meant, and aiming there would aim at a different place than the one that was
reasoned about. Beyond it — or before any floor has been seen — the waypoint is published unchanged.

The model's coordinate is never rewritten. It is what `instruction_plan.json` records, what the
overlay draws, and what the drive loop measures its distance against, because it is what the
scorer grades; the snapped point is an execution detail, recorded per leg as `published` and
`snap_m`.

Then `instruction_reasoner` drives, in **three tries of two distances**:

- `Bool(false)` on `/start_exploration` takes the waypoint topic from TARE;
- a try publishes the waypoint at 2 Hz for `try_duration_s` (15 s). Republishing *is* the
  retry — every message resets the converter's arrival latch and makes it re-pick a
  traversable point;
- come within `settle_radius_m` (1.5 m) **of the model's point**, or within `arrival_m` (0.5 m)
  **of the point actually published**, and the reasoner **stops publishing** and lets the robot
  settle for `settle_s` (5 s), still watching. Another message would reset the latch and restore
  cruise speed, creeping the robot off the pose the goal is scored on;
- otherwise the next try starts, and after `max_tries` (3) the route moves on. At the goal
  there is nothing to move on to, so the robot stops where it is and is scored on that pose.

Worst case is **50 s per waypoint**, so a leg is bounded by construction rather than by a
predicate that has to fire. This replaced an arrival tolerance, a settle window with a
progress predicate, and a nudge/retarget/unwedge ladder — three mechanisms that each run
found a new way to fail to line up, and whose ladder was measured doing nothing: a wedged goal
ran 13 recovery attempts over 149 s while its distance moved 6 cm.

The second test exists because the first cannot always be won. Distance to the model's point is
bounded below by `snap_m` plus the converter's 0.3 m stop deadband, so a leg snapped 1.4 m off can
never enter a 1.5 m circle however well it drives — one measured goal leg burned all three tries and 45.7 s
having arrived 0.28 m from where it was aimed on the first. It is gated on `snap_m > 0`: on a
passthrough the published point *is* the model's point, the two tests collapse into one, and a
wedged leg would report success. `arrival_m` is deliberately tighter than `settle_radius_m` — a
fallback for legs the graded test cannot win, not a cheaper way to pass one it could. Snapping to
the *closest* floor rather than to eroded floor should make it rare, since `snap_m` no longer
starts near the radius; it stays because nothing guarantees that.

`settle_radius_m` **is** the scoring radius. A leg that calls itself arrived while outside the
circle it is graded on has not arrived in any sense that counts. It was 2 m until an earlier
version of this file justified that with "a tolerance derived from the scoring circle was met 0
times in 17 goal legs" — a figure that turned out to be an artifact of `closest_m` being
sampled *before* the settle. Measured on where the robot actually stopped, **8 of those 17 goal
legs were already inside 1.5 m**, and across all 40 legs the median final distance was 1.42 m,
not the 1.85 m the record claimed.

Publishing, not settling, is what closes distance: measured over that sweep, a leg closes
**+0.479 m/s while publishing** and **+0.049 m/s while settling**, and only 3 of 38 publish
windows were frozen. Republishing is doing the driving, which is why the retry is simply more
of it.

`theta` is always 0.0 and is ignored: the deployed converter runs `yawConfig -1`, which holds
the current heading on arrival, and the README says to neglect heading this year.

## Reading a failure afterwards

`instruction_plan.json`, written beside the crops **before the robot moves** (the harness tears
the pipeline down the moment the question ends), records the table sent, the `view_source` and
the image paths, the raw reply, every correction `parse_route` applied, and the final route. A
`drive` block is appended when the route finishes: per leg, where it was last `published` and
how far that is from the model's point (`snap_m`), how many times it re-aimed (`snaps`), which
test ended it (`arrived_at`: `goal`, `target`, or null when the tries ran out), the tries spent,
`closest_m` — which now spans the settle — and two final distances from the pose the robot
**ended** at: `final_m` to the model's point, which is what a goal is scored on, and
`final_target_m` to where it was aimed.

Read those two together. `final_m ≈ snap_m + final_target_m` by construction, so a large
`final_m` with a small `final_target_m` is a snap that had to move far, not a drive that failed
— the robot went exactly where it was sent. `snaps` distinguishes a re-snap that fired and found
nothing from one that never triggered; `snap_m` of 0 with `snaps` above 1 means the terrain
around that waypoint never came into view at all.

| symptom | reading |
|---|---|
| `target_coverage.labels_unseen` non-empty | perception — the object was never detected |
| the plan cites the wrong id | reasoning — the model picked the wrong object |
| the plan is right, the trajectory never reaches it | driving — a stall, the deadline, or the trav-adjust |
| large `final_m`, small `final_target_m` | not driving — the snap had nowhere closer to stand |
| `snap_m` 0 with `snaps` > 1 | no terrain reading near that waypoint on any pass |
| waypoints out of order against the sentence | parsing — the model mis-segmented the command |

`plan_source` says which produced the route: the backend name, or `fallback` when the model
call failed and the reasoner drove the prompt objects in order instead. A fallback route is
wrong about the order roughly as often as it is right — it exists because a robot that drives
somewhere plausible can still satisfy a constraint, and one that never moves scores zero.

## Related

- [cat3_benchmark.md](cat3_benchmark.md) — the ground truth and how it was recovered
- [M5_instruction_planner.md](M5_instruction_planner.md) — the milestone
- `scripts/eval/score.py::score_instruction` — the scorer, our proxy for the organizers'
