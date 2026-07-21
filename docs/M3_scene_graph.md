# M3 — Scene graph

**Task:** Turn the instance memory into a queryable spatial-relation store. Concretely: pairwise relations (near, between, closest-to, on, above/below, viewpoint-relative left/right) over 3D boxes — adapt SORT3D; region/room inference (object clustering + wall layout from lidar); query API for the reasoner; incremental updates as M2 adds instances.

**Status:** ⚪ not started
**Owner:**
**Depends on:** M2

## Interfaces
- In: instance stream from M2
- Out: query API to M4, e.g. `filter(label=chair, color=blue, relation=between(table, wall))` → ranked instances + confidence; full graph JSON dump for LLM context

## Plan
| Stage | What to try |
|---|---|
| Baseline | On-demand relation computation (only for question-relevant instances) using SORT3D's relation definitions/thresholds |
| Upgrade | Region/room inference (doorway detection + object clusters → "kitchen", "living room"); incremental relation cache; support-surface logic for "on" |
| Stretch | View-on-Graph-style multi-layer graph with stored viewpoints per node (lets a VLM traverse and verify) |

## Relation vocabulary (from challenge question style)
`near`, `between(A, B)`, `closest_to(anchor)` / `farthest_from(anchor)`, `on(surface)`, `next_to`, `in(region)`, `left/right of (from viewpoint)`. Calibrate thresholds (what distance = "near"?) on the 75 training questions — the question set defines the semantics, not our intuition.

## Metrics (via M6)
- Relation accuracy on training-question anchors (does "closest to the fridge" pick the GT object?)
- Query latency (<100 ms target)

## Design notes
- "Between" is the tricky one: use the corridor/band between the two anchor boxes, not just line-segment distance.
- "Closest/farthest" are superlatives over a filtered set — compute after attribute filtering, order matters.
- Regions matter for multi-room scenes and for M1's relevance-scored exploration.

## Progress checklist
- [ ] SORT3D vendored/ported; relation functions unit-tested on synthetic boxes
- [ ] Query API serving filter+superlative queries
- [ ] Thresholds calibrated on training questions
- [ ] Region inference on a multi-room scene
- [ ] Graph JSON export (compact, LLM-friendly)

## Suggestions
- Keep the graph tiny in LLM context: only instances matching question nouns + their anchors, not all 200 objects.
- Property-test relations with hand-built toy scenes (3 boxes) before real data — threshold bugs look like grounding bugs downstream.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created |
