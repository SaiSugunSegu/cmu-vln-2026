#!/usr/bin/env python3
"""Bulk pre-seed of the Hugging Face cache.

Nothing forces offline mode — the deployment always has connectivity. This exists so the
first real run is not also a ~20 GB download, and to warm SAM 3's cv-utils kernel, which is
otherwise fetched lazily and fails silently (see warm_kernels).

Run it once after `just up`, and again on any new machine:

    just hf-fetch                 # all models + kernels
    just hf-fetch qwen3vl sam3    # a subset

`facebook/sam3` is a gated repo — it needs HF_TOKEN (repo-root .env) AND the
licence accepted on the model page by the account that issued the token.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Keep these in sync with the defaults at each call site — a drift here shows up as a
# surprise download of a second copy much later, not as an error:
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
    # Ships sam3.1_multiplex.pt (a NATIVE checkpoint) and no safetensors, so
    # from_pretrained cannot load it directly — snapshot_download is only step one.
    # `just sam31-convert` turns it into an HF directory. See docs/M2_perception.md 4.5.
    "sam3.1": {
        "repo_id": "facebook/sam3.1",
        "gated": True,
        "why": "sam_mapper Object Multiplex upgrade — needs `just sam31-convert` after this",
    },
}

ALL_MODELS = {**MODELS, **OPTIONAL_MODELS}


def warm_kernels() -> bool:
    """Pre-seed the kernels-community/cv-utils kernel.

    It downloads lazily on first use and its absence is SILENT: mask NMS is skipped entirely
    (making det_nms_thresh dead) and hole filling stops. Fetched through transformers' own
    loader so the cached revision and API major are the ones it asks for.
    """
    print("\n[hf-fetch] kernels (SAM 3 mask NMS + mask post-processing)", flush=True)
    ok = True

    try:
        from transformers.models.sam3_video import modeling_sam3_video as m

        m._load_cv_utils_kernel_once()
        if getattr(m, "cv_utils_kernel", None):
            print("[hf-fetch] ok -> cv-utils (NMS, hole filling, sprinkle removal)")
        else:
            ok = False
            print("[hf-fetch] cv-utils did NOT load. Mask NMS is skipped silently and "
                  "det_nms_thresh\n"
                  "           becomes a dead knob. Two causes, in order of likelihood:\n"
                  "           1. the `kernels` package version is outside the window "
                  "transformers accepts\n"
                  "              (KERNELS_MIN/MAX_VERSION) — it then reports "
                  "'not installed' even though\n"
                  "              `import kernels` works;\n"
                  "           2. no prebuilt variant for this torch/CUDA — see the pin note "
                  "in\n"
                  "              docker/requirements_captioner.txt.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — never fail setup over an optional kernel
        ok = False
        print(f"[hf-fetch] cv-utils probe failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # flash-attn2 is deliberately not warmed: 447 MB for a kernel we do not use — it
    # detects nothing on SAM 3, so both configs request sdpa.

    print("[hf-fetch] confirm on the target machine with:  just sam-profile "
          "/data/bags/_frames\n"
          "           it prints `cv-utils kernel: ...` and `[sam3] attn effective: ...`")
    return ok


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
        # cannot write. The init one-shot in docker/compose_gpu.yml is what fixes it.
        print(f"[hf-fetch] {repo_id} failed writing the cache: {exc}\n"
              f"           If this is a permission error, run `just up` so the init\n"
              f"           container can chown {os.environ.get('HF_HOME', '~/.cache/huggingface')}.",
              file=sys.stderr)
        return False

    print(f"[hf-fetch] ok -> {path}", flush=True)
    return True


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-seed the HF cache so the first real run is not a 20 GB download.")
    parser.add_argument(
        "models", nargs="*", default=None,
        help=f"Subset to fetch. Default: {' '.join(MODELS)}. "
             f"Also available: {' '.join(OPTIONAL_MODELS)}.")
    parser.add_argument("--list", action="store_true",
                        help="Show the model list and exit without downloading.")
    parser.add_argument("--no-kernels", action="store_true",
                        help="Skip pre-seeding SAM 3's cv-utils kernel. Fetched lazily "
                             "otherwise, and its absence degrades silently.")
    args = parser.parse_args(argv)

    if args.list:
        for name, spec in ALL_MODELS.items():
            default = "default" if name in MODELS else "optional"
            print(f"{name:10s} {spec['repo_id']:35s} ({default}) {spec['why']}")
        return 0

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

    # Weights alone are not enough to run SAM 3 at full quality.
    kernels_ok = True if args.no_kernels else warm_kernels()

    print()
    if failed:
        print(f"[hf-fetch] FAILED: {', '.join(failed)} "
              f"({len(selected) - len(failed)}/{len(selected)} succeeded)",
              file=sys.stderr)
        return 1
    print(f"[hf-fetch] all {len(selected)} model(s) cached.")
    # Not a hard failure, but never return 0 as if it were fine.
    if not kernels_ok:
        print("[hf-fetch] WARNING: cv-utils unavailable — the pipeline runs, but without "
              "mask NMS,\n           and nothing at runtime will say so.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
