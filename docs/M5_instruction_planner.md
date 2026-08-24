# M5 — Instruction-following planner

**Task:** Turn "take the path near X, avoid Y, end at Z" into a scored trajectory. Concretely: take grounded constraint objects from M4 with ordering (pass-near list, avoid list, goal); build a costmap over the terrain map (attraction near pass-constraints, repulsion in avoid zones); generate an ordered, densified waypoint sequence; publish sequentially with progress monitoring; replan if the executed path violates an avoid zone.

**Status:** 🟡 in progress — VLM route planner landed, first live-sim eval pending
**Owner:**
**Depends on:** M4 (grounding), M1 (map)
**Worth the most points: 6/question — invest accordingly (full W4).**

## Interfaces
- In: ordered constraint spec from M4 `{pass_near: [inst...], avoid: [region...], goal: inst}`, terrain map, odometry
- Out: `/way_point_with_heading` (Pose2D sequence, published as each is reached)

## Plan
| Stage | What to try |
|---|---|
| Baseline | ~~One waypoint near each pass-constraint, in order, then goal~~ — **shipped, but the constraints come from a VLM rather than a solver.** One call sees the command, the whole `obj_map.json` as a table, and the best-view frames with every mapped object tagged by its map id; it returns the ordered waypoints. `cat3_utils` validates the coordinates against the rows the model cited, densifies, and drives. See [cat3_vlm_contract.md](cat3_vlm_contract.md) |
| Upgrade | Densified trajectory (**shipped**: 3.5 m hops, sized to the waypoint converter's own 5 m re-target radius rather than the 1 m originally guessed); execution monitor (**shipped**: stall ladder nudge → retarget → unwedge → skip, with each waypoint given a share of the route deadline). Still open: costmap with repulsion for avoid zones, replan on violation |
| Stretch | NavOne-style one-shot global plan on top-down map; trajectory scoring self-check before execution (simulate against our own scorer) |

## Scoring model (design against this)
Score based on: constraints achieved, in correct order, penalties for forbidden areas and wrong order. Partial credit exists. Therefore: (1) order is sacred — never reorder for path efficiency; (2) a longer safe path beats a shorter risky one; (3) always publish something — even goal-only earns partial credit.

## Metrics (via M6)
- Trajectory score vs .ply GT trajectories on training instruction questions
- Avoid-zone violation rate (target 0)
- Constraint-order correctness %

## What the base autonomy actually does with a waypoint

Read `waypoint_converter/src/waypointConverter.cpp` before tuning anything here — most of the
design above follows from it rather than from us:

| constant | value | consequence |
|---|---|---|
| `adjDisThre` / `waypointTravAdj` | 5.0 / true | within 5 m it **replaces** our waypoint with the nearest traversable point |
| `obstacleDisThre` | 0.75 | it never picks a point that close to an obstacle, so the robot parks about a metre short of an object centre — which is where the demos stop too |
| `waypointXYRadius` | 0.3 | arrival latches against the *adjusted* point, then commands speed 0 |
| `/way_point_reached` | `Float32` | the residual to the waypoint we asked for: a better arrival test than our own distance |
| `waypointHandler` | resets the latch | republishing is required while driving, and must **stop** on arrival or the robot creeps off the scored pose |
| `yawConfig` | −1 | holds current heading on arrival, so our `theta` is ignored — consistent with the README's "neglect the heading" |

## Design notes
- **Waypoints must be CLOSE to the vehicle** — organizer's README: far waypoints can wedge the local planner in dead ends. Chop every path into ≤2–3 m hops; for distant goals consider vendoring FAR planner (visibility-graph global planning, already in the organizer's extended stack). `waypoint_example` node shows the canonical publish pattern (waypoint + boundary + speed).
- Base autonomy adjusts out-of-traversable waypoints — dense waypoints keep the executed path under OUR control, not the planner's.
- "Path near the window" = the waypoint band along the window's traversable side; sample 2–3 waypoints there, not 1.
- Avoid regions from "between the two tables": construct the polygon between anchor boxes, inflate by robot radius.
- Wait-for-arrival logic: publish next waypoint when within r of current one.

## Progress checklist
- [x] Constraint spec schema agreed with M4 (Aug 23 — there is no M4/M5 boundary any
      more: one VLM call goes from the command straight to ordered waypoints over the
      robot's own map. `RoutePlan` in `vlm_backends/schemas.py` is the schema)
- [x] Baseline ordered waypoints on a training instruction question (Aug 23 — landed;
      first live-sim eval pending)
- [x] Own trajectory scorer (M6) replicating constraint/order/penalty logic vs .ply GT
      (Aug 14 — GT for all 30 instruction questions in `data/benchmark/<scene>/category_3/`;
      `just cat3-verify` replays every provided .ply through `score_instruction` and requires
      6/6. See [cat3_benchmark.md](cat3_benchmark.md). The constraint spec there —
      ordered `pass_near` / `avoid` / `goal` over IRef-VLA object ids — is a ready-made
      candidate for the M4↔M5 schema above.)
- [ ] Costmap + avoid-zone repulsion
- [x] Densification + execution monitor (Aug 23 — 3.5 m hops, stall ladder, route deadline)
- [ ] Replan-on-violation tested (force a violation)
- [ ] All training instruction questions scored; failures categorized

## Suggestions
- Build our own scorer FIRST (from the scoring description + .ply files) — iterating blind on 6-point questions is the biggest schedule risk in the project.
- Visualize planned path vs executed path vs GT .ply in one RViz/Foxglove view — makes failure modes obvious in seconds.
- Study the provided .ply target trajectories: they reveal what the organizers consider "near" and how tight the corridors are.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created |
| Aug 14 | Category-3 GT landed for all 15 scenes, recovered from the organizers' demo .ply trajectories. Key finding: the trajectory frame and the IRef-VLA annotation frame are identical, so demo and boxes compare directly. `just cat3-verify` — 29/30 at 6/6. [cat3_benchmark.md](cat3_benchmark.md) |
| Aug 23 | Planner landed as a single VLM call: command + `obj_map.json` + tagged best views -> ordered waypoints, validated against the cited map rows and driven with a stall ladder and a route deadline. No solver, and no geometry shared with the GT generator, so the benchmark stays an independent check. Budget is now category-aware (T-210 for instruction questions; the old T-90 was sized for a model call, not a 33 m drive). Contract: [cat3_vlm_contract.md](cat3_vlm_contract.md) |
| Aug 15 | Category-3 GT now carries the *reference* objects too — the nouns an instruction uses to say which target it means ("the stool **under the picture**"). 94 constraint boxes + 60 reference boxes, each tagged `role`, with `anchor_ids` per constraint. `target_objects` is complete, so it is usable as the SAM-3 prompt list. Scored geometry unchanged; gate still 29/30 at 6/6. |
