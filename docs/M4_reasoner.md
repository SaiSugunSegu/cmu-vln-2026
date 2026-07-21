# M4 — Reasoner / orchestrator

**Task:** Own the question lifecycle from parse to answer. Concretely: parse `/challenge_question` → type + target spec (objects, attributes, relations, ordering); run the explore/answer decision gate on each graph update; ground via symbolic queries over M3 (query-graph → subgraph matching); trigger `reobserve()` on ambiguity; verify borderline cases by sending stored crops to a VLM; publish the answer on the right topic; enforce the T-30s fallback.

**Status:** ⚪ not started
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
- [ ] T-30s fallback tested (kill exploration mid-run, must still answer)
- [ ] Provider fallback path tested

## Suggestions
- Cache LLM parses: the 75 training questions are fixed — parse once, iterate on grounding cheaply.
- Log every reasoner decision (query, candidates, scores) to a per-run JSON — postmortems on wrong answers are impossible without it.
- Try the parse step with a small local model (e.g., Qwen-class) early; if it matches API quality, one less network dependency.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created |
