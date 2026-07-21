# Contributing / team workflow

## Repo
One codebase, three remotes on the same local clone:

| Remote | Repo | Visibility | Purpose |
|---|---|---|---|
| `origin` | `SaiSugunSegu/cmu-vln-2026` | **private** | Daily work: code + docs. Push here. |
| `upstream` | `Yuxin916/CMU-VLN-Challenge-2026` | public | Organizer updates: `git fetch upstream && git merge upstream/main` |
| `fork` | `SaiSugunSegu/CMU-VLN-Challenge-2026` | public | **Submission only.** Untouched during dev (push URL disabled); at submission: re-enable + `git push fork main` |

Setup (already done on the L4 box / see main clone):
```bash
git remote rename origin fork
git remote set-url --push fork DISABLED
git remote add origin git@github.com:SaiSugunSegu/cmu-vln-2026.git
git remote add upstream https://github.com/Yuxin916/CMU-VLN-Challenge-2026.git
git push -u origin main
```

Still: **never commit API keys** — `.env` (gitignored) + env vars; per challenge FAQ, keys ship inside the submitted Docker image, not the repo. Never modify upstream files — our changes live only in `ai_module/` + docs folders, keeping upstream merges clean.

## Workflow
- Branch per task: `m2/sam3-bakeoff`, `m5/costmap`, `docs/...`
- PR into `main`; before merge:
  1. M6 smoke subset run — did the score go up (or stay flat for refactors)?
  2. Component doc updated (checklist ticks, log entry) in the same PR.
  3. New experiment? Add `experiments/YYYY-MM-DD_short-name.md` from the template.
- Decisions that affect other modules go in the README decision log.
- Big model files: never commit — use the shared drive / HF hub; `.gitignore` covers common paths.

## Environment ground rules
- Everything runs in the two challenge containers; no host-installed deps beyond Docker + NVIDIA toolkit.
- Any package added to the AI module image goes in its Dockerfile (pinned version), not installed ad hoc — eval rebuilds our image.
- Bags for offline dev: record per M0 runbook, store under shared drive `bags/<scene>/`, referenced (not committed) in experiments.

## Question hygiene
Questions to organizers: open a GitHub issue with the "question" label on the challenge repo (whole team sees answers), or email jingfant@andrew.cmu.edu.
