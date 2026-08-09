# M1 — Exploration

**Task:** Get the robot to see every object fast. Coverage-guaranteed exploration + an API for the reasoner: `explore_next()`, `goto(x, y)`, `reobserve(instance_id)` (pose with clear line of sight to an object), plus a "coverage done" signal.

**Status:** 🟡 in progress (supervisor skeleton ✅ Jul 22; TARE vendoring next)
**Owner:**
**Depends on:** M0

## Interfaces
- In: `/terrain_map_ext` (occupancy source), `/state_estimation`, requests from M4
- Out: `/way_point_with_heading` (Pose2D, ≤2.5 m hops), coverage % + status to M4

## DECISION (Jul 22): TARE engine + supervisor on top

TARE (already in the organizer's stack) drives coverage; our `smart_vlm` supervisor owns every scoring decision — it gates when exploration starts (`/pipeline/armed`) and when it ends (`/pipeline/explore_done`, on bag end, explore timeout, or T-90), and guarantees an answer at T-30. Files: `ai_module/src/smart_vlm/` (supervisor + launch), `scripts/challenge_simulation.sh` (domain-firewall eval mimic). Vendoring steps: comments in `smart_vlm.launch` (submodule init → copy `tare_planner` into ai_module → verify build in AI container).

Rationale vs alternatives: benchmarks favor TARE over GBPlanner-class planners for ground-robot indoor work, but at room scale ALL planners converge — the ecosystem match (same CMU stack, pre-integrated) is the real argument. Fallback if vendoring fights us: sparse viewpoint sweep — the 360° camera sees a full disk per pose, so "observed" = within ~4 m of a past pose; single rooms ≈ 3–6 viewpoints; nearest-frontier over that, optionally mini-TSP (5–10 nodes) for route quality.

**The supervisor IS the winning mechanism** (TARE only knows "cover space"):
- *Mission clock*: 600 s budget → answer-mode at T-90 (stop exploring) → fallback publish at T-30 (never a silent zero).
- *Question-conditioned stopping*: numerical = full sweep mandatory (missed object = wrong count); object-ref = stop at confident grounding + margin over distractors (→ tiebreaker bonus); instruction = stop when referenced objects grounded + path corridor mapped.
- *Waypoint mux*: exactly ONE writer to `/way_point_with_heading` — supervisor forwards TARE during sweep, overrides with `reobserve(id)` poses (1.5–3 m range, LoS-verified, ≥60° novel angle), hands channel to M5 for instruction execution. TARE without a mux will fight M5 — the #1 integration bug to avoid.
- *Object-coverage done-signal*: sweep ends when instances have ≥2 views ≥60° apart — floor % is a proxy, object coverage is the requirement.
- *Question-biased visit order* (upgrade): SigLIP-scored frontiers + lidar doorway detection front-load the relevant region in multi-room scenes; doesn't reduce total coverage, enables earlier stopping.

## Topic → usage map (all 6 allowed inputs)
| Topic | Used by | For |
|---|---|---|
| /camera/image (5 Hz measured) | M2 | detection/segmentation crops; re-ID crops; captions |
| /registered_scan | M2, M3 | accumulated voxel map (map frame) → mask-projected 3D boxes |
| /sensor_scan | (spare) | fallback for tight image↔scan sync; unused if odometry interpolation suffices |
| /terrain_map (5 m) | M5 | fine local costmap: avoid-zone shaping |
| /terrain_map_ext (20 m) | M1, M5 | free-space grid, viewpoint planning, TARE input, global costmap |
| /state_estimation (200 Hz) | M1, M2, M5 | coverage tracking; image-time pose interpolation; execution monitoring |
| /challenge_question (in) | supervisor | 1 Hz repeated — latch first |

## Metrics (via M6)
- Time to 95% object-coverage (target <3 min single room, <6 min multi-room)
- % GT objects observed ≥2× from ≥60° apart
- Reobserve success rate; answer-time distribution (tiebreaker data)

## Progress checklist
- [x] Supervisor skeleton: question latch + classify, mission clock, topic health, output pubs (Jul 22)
- [ ] TARE vendored into ai_module; builds in AI container; explores autonomously in sim
- [ ] TARE start/stop control from supervisor confirmed (check `system_simulation_with_exploration_planner.sh` for the kick-off mechanism)
- [ ] Waypoint mux: single writer, reobserve override working
- [ ] Coverage tracker (observed = 4 m disk around trail; calibrate radius on a bag)
- [ ] Object-coverage done-signal wired to M2 instances
- [ ] `reobserve(id)` pose computation with LoS check
- [ ] Question-conditioned stopping policy
- [ ] Multi-room scene: doorway handling, covered <6 min

## Notes
- Waypoints must be near the vehicle (organizer README: far waypoints wedge the local planner) — ≤2.5 m hops everywhere.
- Base autonomy auto-adjusts out-of-traversable waypoints — plan conservative poses anyway.
- 360° camera → heading control unnecessary (repo: heading neglected this year).
- Log coverage-vs-time curves from day one — the tiebreaker knob.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created |
| Jul 22 | Decision: TARE-vendored + supervisor; smart_vlm package created (supervisor v0 + launch); challenge_simulation.sh firewall added |
