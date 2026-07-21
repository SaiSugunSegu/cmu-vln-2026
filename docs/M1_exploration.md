# M1 — Exploration

**Task:** Get the robot to see the whole (relevant) scene fast. Build coverage tracking from `/terrain_map_ext` + odometry, extract and score frontier viewpoints, publish `Pose2D` waypoints, and expose an API to the reasoner: `explore_next()`, `goto(x, y)`, `reobserve(instance_id)` (drive to a pose with clear line of sight to an object), plus a "coverage done" signal.

**Status:** ⚪ not started
**Owner:**
**Depends on:** M0

## Interfaces
- In: `/terrain_map_ext` (PointCloud2), `/state_estimation` (Odometry), requests from M4
- Out: `/way_point_with_heading` (Pose2D), coverage % + frontier list to M4

## Plan
| Stage | What to try |
|---|---|
| Baseline | 2D occupancy grid from terrain map; nearest-frontier selection; coverage % = observed cells / traversable cells |
| Upgrade | Language/relevance-scored frontiers (InstructNav/OmniNav-style value map: bias toward regions likely to contain question objects); targeted re-observation poses for ambiguous instances (viewpoint with clear LoS at 1.5–3 m) |
| Stretch | Viewpoint-sequence optimization (mini-TSP over frontiers); learned/calibrated stopping criterion |

## Metrics (via M6)
- Time to 90% coverage per scene (target: <3 min single room, <6 min multi-room)
- % of question-relevant objects observed before answer
- Re-observation success rate (instance visible from chosen pose)

## Design notes
- 360 camera = no need to control heading for viewing; heading matters only for path shape. Repo says heading is neglected this year.
- Waypoints outside traversable area get auto-adjusted by base system — but plan conservative poses anyway; adjustment can teleport intent.
- Multi-room scenes: doorway detection from terrain map (narrow traversable corridors between large free regions) → treat rooms as exploration units.
- Budget manager owns the 10-min clock: exploration gets a decreasing share as time passes; hard handoff to answer mode at T-60s.

## Progress checklist
- [ ] Occupancy/coverage grid node running off bag data
- [ ] Frontier extraction visualized in RViz/Foxglove
- [ ] Nearest-frontier loop explores scene 1 end-to-end
- [ ] Coverage % metric logged; <3 min on a single room
- [ ] `reobserve()` pose computation with LoS check
- [ ] Relevance-scored frontier upgrade
- [ ] Multi-room test scene handled

## Suggestions
- Steal frontier code: `nav2` frontier examples or `explore_lite` (port to Jazzy) rather than writing from scratch.
- Log per-run coverage-vs-time curves early — this is the tiebreaker (early finish bonus) and the main knob when questions time out.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created |
