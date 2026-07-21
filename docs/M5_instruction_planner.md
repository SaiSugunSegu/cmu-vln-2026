# M5 — Instruction-following planner

**Task:** Turn "take the path near X, avoid Y, end at Z" into a scored trajectory. Concretely: take grounded constraint objects from M4 with ordering (pass-near list, avoid list, goal); build a costmap over the terrain map (attraction near pass-constraints, repulsion in avoid zones); generate an ordered, densified waypoint sequence; publish sequentially with progress monitoring; replan if the executed path violates an avoid zone.

**Status:** ⚪ not started
**Owner:**
**Depends on:** M4 (grounding), M1 (map)
**Worth the most points: 6/question — invest accordingly (full W4).**

## Interfaces
- In: ordered constraint spec from M4 `{pass_near: [inst...], avoid: [region...], goal: inst}`, terrain map, odometry
- Out: `/way_point_with_heading` (Pose2D sequence, published as each is reached)

## Plan
| Stage | What to try |
|---|---|
| Baseline | One waypoint near each pass-constraint (offset toward path direction), in order, then goal |
| Upgrade | Costmap w/ repulsion for avoid zones; **densified trajectory** (waypoints every ~1 m so base autonomy can't shortcut through forbidden areas — Beyond Waypoints insight); execution monitor comparing odometry vs plan; replan on violation |
| Stretch | NavOne-style one-shot global plan on top-down map; trajectory scoring self-check before execution (simulate against our own scorer) |

## Scoring model (design against this)
Score based on: constraints achieved, in correct order, penalties for forbidden areas and wrong order. Partial credit exists. Therefore: (1) order is sacred — never reorder for path efficiency; (2) a longer safe path beats a shorter risky one; (3) always publish something — even goal-only earns partial credit.

## Metrics (via M6)
- Trajectory score vs .ply GT trajectories on training instruction questions
- Avoid-zone violation rate (target 0)
- Constraint-order correctness %

## Design notes
- Base autonomy adjusts out-of-traversable waypoints — dense waypoints keep the executed path under OUR control, not the planner's.
- "Path near the window" = the waypoint band along the window's traversable side; sample 2–3 waypoints there, not 1.
- Avoid regions from "between the two tables": construct the polygon between anchor boxes, inflate by robot radius.
- Wait-for-arrival logic: publish next waypoint when within r of current one (dummy VLM shows the pattern).

## Progress checklist
- [ ] Constraint spec schema agreed with M4
- [ ] Baseline ordered waypoints on a training instruction question
- [ ] Own trajectory scorer (M6) replicating constraint/order/penalty logic vs .ply GT
- [ ] Costmap + avoid-zone repulsion
- [ ] Densification + execution monitor
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
