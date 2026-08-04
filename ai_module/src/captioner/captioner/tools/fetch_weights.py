#!/usr/bin/env python3
"""One-time Hugging Face weight download for offline deploy.

Everything else in this repo runs with HF_HUB_OFFLINE=1 (the image default): the
robot has no internet, so every from_pretrained() must hit a pre-seeded cache.
This is the single place that is meant to go online.

Run it once after `just up`:

    just hf-fetch                 # all models
    just hf-fetch qwen3vl sam3    # a subset

`facebook/sam3` is a gated repo — it needs HF_TOKEN (repo-root .env) AND the
licence accepted on the model page by the account that issued the token.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Keep these in sync with the defaults at each call site — a drift here shows up
# as a confusing offline load failure much later:
#   sam3     -> ai_module/src/sam_mapper/config/sam3_*.yaml  ("model_id:")
#   qwen3vl  -> captioner/models/captioning.py  (Qwen3VLHFBackend.default_model_id)
#   qwen2_5vl-> captioner/models/captioning.py  (QwenHFBackend.default_model_id)
#   clip     -> captioner/models/clip.py        (OpenCLIP default model_id)
MODELS: dict[str, dict] = {
    "sam3": {
        "repo_id": "facebook/sam3",
        "gated": True,
        "why": "sam_mapper detection/segmentation",
    },
    "qwen3vl": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "gated": False,
        "why": "captioner + qwen_vqa_server + category-1 (default backend)",
    },
    "clip": {
        "repo_id": "apple/DFN5B-CLIP-ViT-H-14-378",
        "gated": False,
        "why": "captioning_node crop/query matching (OpenCLIP)",
    },
}

# Not fetched by default: an alternate backend, only needed with
# --captioning_model qwen2_5vl.
OPTIONAL_MODELS: dict[str, dict] = {
    "qwen2_5vl": {
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "gated": False,
        "why": "alternate captioning backend",
    },
}

ALL_MODELS = {**MODELS, **OPTIONAL_MODELS}


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def fetch(name: str, spec: dict, token: Optional[str]) -> bool:
    """Download one repo. Returns True on success."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    repo_id = spec["repo_id"]
    print(f"\n[hf-fetch] {name}: {repo_id}  ({spec['why']})", flush=True)
    try:
        # Resumes and skips files already present, so re-running is cheap.
        path = snapshot_download(repo_id, token=token)
    except GatedRepoError:
        print(
            f"[hf-fetch] {repo_id} is GATED and this token cannot read it.\n"
            f"           1. Open https://huggingface.co/{repo_id} and accept the licence\n"
            f"              with the same account that issued HF_TOKEN.\n"
            f"           2. Check HF_TOKEN in the repo-root .env is valid and not expired.",
            file=sys.stderr)
        return False
    except RepositoryNotFoundError:
        print(
            f"[hf-fetch] {repo_id} not found. If it is private or gated this is what\n"
            f"           a missing/invalid HF_TOKEN looks like — check the repo-root .env.",
            file=sys.stderr)
        return False
    except LocalEntryNotFoundError:
        print(
            f"[hf-fetch] {repo_id} could not be reached and is not cached — no network?",
            file=sys.stderr)
        return False
    except OSError as exc:
        # Covers the permission case: a root-owned cache dir the container's uid
        # cannot write. The init one-shot in docker/compose.yml is what fixes it.
        print(f"[hf-fetch] {repo_id} failed writing the cache: {exc}\n"
              f"           If this is a permission error, run `just up` so the init\n"
              f"           container can chown {os.environ.get('HF_HOME', '~/.cache/huggingface')}.",
              file=sys.stderr)
        return False

    print(f"[hf-fetch] ok -> {path}", flush=True)
    return True


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-seed the HF cache so every later run works offline.")
    parser.add_argument(
        "models", nargs="*", default=None,
        help=f"Subset to fetch. Default: {' '.join(MODELS)}. "
             f"Also available: {' '.join(OPTIONAL_MODELS)}.")
    parser.add_argument("--list", action="store_true",
                        help="Show the model list and exit without downloading.")
    args = parser.parse_args(argv)

    if args.list:
        for name, spec in ALL_MODELS.items():
            default = "default" if name in MODELS else "optional"
            print(f"{name:10s} {spec['repo_id']:35s} ({default}) {spec['why']}")
        return 0

    if _offline():
        print(
            "[hf-fetch] HF_HUB_OFFLINE is set — nothing would be downloaded.\n"
            "           This recipe is meant to run with it unset:\n"
            "               just hf-fetch\n"
            "           (which passes HF_HUB_OFFLINE=0 for this command only).",
            file=sys.stderr)
        return 2

    selected = args.models or list(MODELS)
    unknown = [m for m in selected if m not in ALL_MODELS]
    if unknown:
        print(f"[hf-fetch] unknown model(s): {unknown}. "
              f"Choose from: {sorted(ALL_MODELS)}", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN") or None
    if token is None and any(ALL_MODELS[m]["gated"] for m in selected):
        print(
            "[hf-fetch] warning: no HF_TOKEN in the environment; gated repos will "
            "fail.\n           Put HF_TOKEN=hf_… in the repo-root .env.",
            file=sys.stderr)

    failed = [m for m in selected if not fetch(m, ALL_MODELS[m], token)]

    print()
    if failed:
        print(f"[hf-fetch] FAILED: {', '.join(failed)} "
              f"({len(selected) - len(failed)}/{len(selected)} succeeded)",
              file=sys.stderr)
        return 1
    print(f"[hf-fetch] all {len(selected)} model(s) cached. "
          f"Later runs work offline (HF_HUB_OFFLINE=1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
