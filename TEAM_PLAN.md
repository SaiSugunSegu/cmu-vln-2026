# CMU VLN Challenge 2026 — Team Plan

IROS 2026 · Simulation phase · [Challenge site](https://www.ai-meets-autonomy.com/cmu-vln-challenge) · [Challenge repo](https://github.com/Yuxin916/CMU-VLA-Challenge-2026)

Daily work lives in the **private** repo `SaiSugunSegu/cmu-vln-2026` (`origin`); the public fork is used only at submission time (remotes table: [CONTRIBUTING.md](CONTRIBUTING.md)). Our code goes in `ai_module/`; docs in `docs/`, `experiments/`, `scripts/`, and this file — never modify upstream files.

**Architecture diagram + team quickstart: [README.md](README.md)** · **Design & proposals: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## Key dates
| Date | Milestone |
|---|---|
| **Jul 25, 2026 (Sat)** | Registration deadline |
| Aug 25, 2026 | Submission deadline (multiple submissions allowed, best kept) |

## Scoring (per question, 5 questions × 3 test scenes)
- Numerical /1 (exact Int32) · Object reference /2 (bbox overlap) · Instruction following /6 (constraint order + penalties)
- 10 min/question incl. exploration; system relaunches per question (no memory); early finish = tiebreaker bonus.

## Component docs
| Doc | Component | Owner | Status |
|---|---|---|---|
| [docs/M0_infra.md](docs/M0_infra.md) | Docker, sim, ROS I/O, visualization | | 🟢 baseline done |
| [docs/M1_exploration.md](docs/M1_exploration.md) | TARE-vendored exploration + supervisor | | 🟡 in progress |
| [docs/M2_perception.md](docs/M2_perception.md) | Open-vocab detection → 3D instances | | ⚪ not started |
| [docs/M3_scene_graph.md](docs/M3_scene_graph.md) | VLA-3D relations, queryable store | | ⚪ not started |
| [docs/M4_reasoner.md](docs/M4_reasoner.md) | Question parsing, grounding, decision gate | | ⚪ not started |
| [docs/M5_instruction_planner.md](docs/M5_instruction_planner.md) | Constraint-ordered waypoint plans | | ⚪ not started |
| [docs/M6_eval_harness.md](docs/M6_eval_harness.md) | Eval bench (15 scenes × 5 Qs → scores) | | 🟡 v0 untested |

Status legend: ⚪ not started · 🟡 in progress · 🟢 baseline done · 🔵 upgrade done · ✅ frozen

## Weekly milestones
| Week | Theme | What to try | Exit criteria |
|---|---|---|---|
| **W1** Jul 21–27 | It runs | ~~M0 setup~~ ✅ · Register · smart_vlm skeleton ✅ · TARE vendored + exploring · bags for 3 scenes · bench smoke run w/ dummy · SAM 3 vs YOLOE bake-off on Unity renders | Robot explores a room autonomously <3 min; bench scores dummy; perception model picked w/ L4 latency |
| **W2** Jul 28–Aug 3 | Baseline e2e | Walking skeleton (ARCHITECTURE §3): parse → explore → perceive → answer, all 3 types; first full 75-Q run | Non-zero score on every question type; instance recall ≥70%, dupes <15% |
| **W3** Aug 4–10 | It answers | SAM 3 + SigLIP 2 re-ID; VLA-3D relations ported; tool-calling agent + decision gate + reobserve | ≥70% numerical + object-ref on training Qs; dupes <10% |
| **W4** Aug 11–17 | It follows | M5 costmap + dense waypoints + monitor; bench trajectory scorer; multi-room exploration; model-watch recheck | Majority of instruction Qs >3/6 on our scorer |
| **W5** Aug 18–25 | It survives | Timing/early-stop tuning; forced fallback tests; API retries + 2nd provider; clean amd64 rebuild | Zero unanswered questions on full run; **submit ~Aug 20**, iterate to Aug 25 |

## Standing rules
1. Always emit an answer before timeout (T-30s fallback) — partial credit is free points.
2. No model choice locked without a latest-version check; recheck in W4 before freeze.
3. Final Docker image must be linux/amd64 (eval NUC is x86_64).
4. Every change is measured by the bench before merging; every experiment gets an `experiments/` entry.
5. Docs updated in the same PR as the code they describe.

## Decision log
| Date | Decision | Why |
|---|---|---|
| Jul 21 | Modular pipeline over end-to-end VLA | Waypoint interface + exact counting favor symbolic graph answers |
| Jul 21 | SAM 3 primary perception candidate | Single model: open-vocab detect + segment + track |
| Jul 21 | L4 box (i9-13900 + L4 24GB) as sole dev machine, single-box | Mirrors eval NUC (same i9 class, x86_64, Ubuntu 24.04) |
| Jul 22 | M3 relations = VLA-3D heuristics ported verbatim | The exact functions that generated the questions define the answer key |
| Jul 22 | M1 = TARE (vendored into ai_module) + supervisor node on top | TARE ships in organizer's stack; supervisor owns clock/stopping/mux — the scoring logic |
| Jul 22 | Eval-mimic via ROS domain firewall (`scripts/challenge_simulation.sh`) | Restriction is on what we CONSUME; domain bridge makes cheating physically impossible |
| | | |
