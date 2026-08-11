#!/usr/bin/env python3
"""Convert facebook/sam3.1's native checkpoint into a loadable HF directory — and prove it.

3.1 ships sam3.1_multiplex.pt and no safetensors, so from_pretrained fails after parsing
config.json: the architecture is supported, only the weight format is wrong
(docs/M2_perception.md 4.5). Once converted, sam_mapper needs no code change — just point
sam3.model_id at the output.

Three hazards this handles:

1. The converter is not in the wheel and not at any release tag — it exists only on `main`
   (checked 2026-08-11 against v5.14.0/v5.14.1/v5.15.0). We pin and report a COMMIT SHA.
2. It calls load_state_dict(strict=False) and was written for SAM 3. If Object Multiplex
   renamed tensors they stay RANDOMLY INITIALISED and the model loads, runs and segments
   worse in silence. Log scraping cannot catch it (transformers sets propagate=False), so we
   wrap load_state_dict and read its return value.
3. Not every missing key is a bug — the converter documents four benign cases, so those are
   allowlisted (BENIGN_MISSING) and only the rest fail.

Two further substitutions the converter makes are undone here, because both would change
behaviour without failing the key audit: it builds the config from a hardcoded SAM 3 default
(we use 3.1's config.json) and the tokenizer from openai/clip-vit-base-patch32
(we restore 3.1's own — see _sync_repo_processor).

    python3 /home/docker/scripts/eval/convert_sam31.py --out /data/models/sam3.1_hf
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Missing keys the converter's own source calls expected (see its comment block right after
# the load_state_dict call). Matched as regexes against full parameter names.
BENIGN_MISSING = (
    r"patch_embeddings\.projection\.bias$",      # patch projection has bias=False upstream
    r"geometry_encoder\.mask_encoder\.projection\.",   # nn.Identity() in the original
    r"rotary_emb\.rope_embeddings$",             # precomputed upstream, on-the-fly here
    r"text_projection\.bias$",                   # projection may have no bias
)

CONVERTER_PATH = "src/transformers/models/sam3_video/convert_sam3_video_to_hf.py"


def _classify(keys, patterns=BENIGN_MISSING):
    benign, suspicious = [], []
    for key in keys:
        (benign if any(re.search(p, key) for p in patterns) else suspicious).append(key)
    return benign, suspicious


def _fetch_converter(ref: str, dest: str) -> str:
    """Download the converter, resolving a branch to a sha first — a conversion you cannot
    re-run against the same converter is not a result you can defend."""
    import json
    import urllib.request

    sha = ref
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        api = f"https://api.github.com/repos/huggingface/transformers/commits/{ref}"
        with urllib.request.urlopen(api, timeout=30) as response:
            sha = json.load(response)["sha"]
        print(f"[convert] ref {ref!r} resolves to commit {sha}")

    url = f"https://raw.githubusercontent.com/huggingface/transformers/{sha}/{CONVERTER_PATH}"
    print(f"[convert] fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read()
    if len(source) < 1000:
        raise SystemExit(f"[convert] converter download looks empty ({len(source)} bytes) — "
                         f"has the file moved? Check {CONVERTER_PATH} at {sha}.")
    with open(dest, "wb") as handle:
        handle.write(source)
    print(f"[convert] converter: {len(source)} bytes -> {dest}")
    return sha


def _load_module(path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("convert_sam3_video_to_hf", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as err:
        raise SystemExit(
            f"[convert] the converter could not import against the installed transformers:\n"
            f"          {type(err).__name__}: {err}\n"
            f"          The converter tracks main; the installed transformers is older. Pin an\n"
            f"          older --converter-ref, or upgrade transformers.") from err
    return module


def _find_checkpoint(repo_id: str, explicit: str | None) -> str:
    """Locate the native .pt in the HF cache (downloading is `just hf-fetch sam3.1`'s job)."""
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"[convert] --checkpoint {explicit} does not exist")
        return explicit

    from huggingface_hub import snapshot_download

    try:
        root = snapshot_download(repo_id, local_files_only=True)
    except Exception as err:  # noqa: BLE001 — every hub error means the same thing
        raise SystemExit(
            f"[convert] {repo_id} is not in the local HF cache ({type(err).__name__}).\n"
            f"          Fetch it first (this is the repo's only online step):\n"
            f"              just hf-fetch sam3.1") from err

    candidates = sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt"))
    if not candidates:
        raise SystemExit(f"[convert] no .pt checkpoint in {root} — contents: "
                         f"{sorted(os.listdir(root))}")
    if len(candidates) > 1:
        print(f"[convert] several .pt files, taking the first: {candidates}")
    return candidates[0]


# Files 3.1 ships that the converter overwrites with its own substitutes.
REPO_PROCESSOR_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                        "merges.txt", "special_tokens_map.json", "processor_config.json")


def _sync_repo_processor(snapshot: str, out: str) -> None:
    """Restore the repo's own tokenizer/processor files over the converter's.

    The converter hardcodes CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    and a 1008x1008 processor, discarding what facebook/sam3.1 ships. If 3.1's tokenizer
    differs, prompts tokenize differently and detections degrade — with no missing tensor for
    the key audit to catch.
    """
    import filecmp
    import shutil

    replaced, identical = [], []
    for name in REPO_PROCESSOR_FILES:
        src = os.path.join(snapshot, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(out, name)
        if os.path.isfile(dst) and filecmp.cmp(src, dst, shallow=False):
            identical.append(name)
            continue
        shutil.copy2(src, dst)
        replaced.append(name)

    if identical:
        print(f"[convert] processor files identical to the repo's: {', '.join(identical)}")
    if replaced:
        print(f"[convert] RESTORED from {snapshot} (converter's differed): "
              f"{', '.join(replaced)}")


def _audit(missing, unexpected, strict_unexpected: bool) -> int:
    """-> process exit code. The whole point of the script."""
    benign, suspicious = _classify(missing)

    print("\n" + "=" * 72)
    print(f"[audit] load_state_dict returned {len(missing)} missing, "
          f"{len(unexpected)} unexpected keys")
    if benign:
        print(f"\n[audit] {len(benign)} missing keys are the documented-benign kind "
              f"(no weights exist upstream):")
        for key in benign[:10]:
            print(f"          - {key}")
        if len(benign) > 10:
            print(f"          ... and {len(benign) - 10} more")

    ok = True
    if suspicious:
        ok = False
        print(f"\n[audit] !! {len(suspicious)} MISSING keys are NOT accounted for. These "
              f"tensors are RANDOMLY INITIALISED in the saved model:")
        for key in suspicious[:40]:
            print(f"          - {key}")
        if len(suspicious) > 40:
            print(f"          ... and {len(suspicious) - 40} more")
        print("\n          This is the Object Multiplex renaming trap from "
              "docs/M2_perception.md 4.5.\n"
              "          The output would load and run and segment WORSE, with nothing in\n"
              "          its behaviour saying so. Do not point sam3.model_id at it.")

    if unexpected:
        print(f"\n[audit] {len(unexpected)} UNEXPECTED keys — present in the checkpoint, "
              f"consumed by nothing:")
        for key in unexpected[:40]:
            print(f"          - {key}")
        if len(unexpected) > 40:
            print(f"          ... and {len(unexpected) - 40} more")
        print("\n          For SAM 3.1 specifically, unexpected keys are the OTHER half of\n"
              "          the same trap: Object Multiplex weights the converter's key map\n"
              "          does not know about get dropped on the floor, and what you end up\n"
              "          running is SAM 3 wearing a 3.1 filename.")
        if strict_unexpected:
            ok = False

    print("=" * 72)
    if not ok:
        print("[audit] FAILED — conversion is not trustworthy.")
        return 1
    print("[audit] PASSED — every tensor the model needs came from the checkpoint.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="facebook/sam3.1",
                       help="HF repo holding the native checkpoint and config.json")
    parser.add_argument("--checkpoint", default=None,
                       help="path to the native .pt (default: find it in the HF cache)")
    parser.add_argument("--out", default="/data/models/sam3.1_hf",
                       help="output HF model directory — put sam3.model_id here when it passes")
    parser.add_argument("--converter-ref", default="main",
                       help="transformers branch or commit sha to take the converter from. "
                            "No RELEASE TAG carries this file; main is the only source.")
    parser.add_argument("--converter", default=None,
                       help="use an already-downloaded converter instead of fetching one")
    parser.add_argument("--strict-unexpected", action="store_true",
                       help="also fail when the checkpoint carries keys the model ignores "
                            "— the right setting when adopting 3.1 for real")
    parser.add_argument("--sync-processor", action=argparse.BooleanOptionalAction, default=True,
                       help="restore the repo's own tokenizer/processor files over the "
                            "converter's hardcoded CLIP substitutes (default: on)")
    parser.add_argument("--config-from-repo", action=argparse.BooleanOptionalAction, default=True,
                       help="build the config from the repo's own config.json rather than the "
                            "converter's hardcoded SAM 3 default (default: on)")
    args = parser.parse_args(argv)

    import torch
    import transformers

    print(f"[convert] transformers {transformers.__version__}, torch {torch.__version__}")

    checkpoint = _find_checkpoint(args.repo, args.checkpoint)
    print(f"[convert] checkpoint: {checkpoint} "
          f"({os.path.getsize(checkpoint) / 1e9:.2f} GB)")

    os.makedirs(args.out, exist_ok=True)
    converter_path = args.converter or os.path.join(args.out, "_converter.py")
    sha = args.converter_ref if args.converter else _fetch_converter(args.converter_ref,
                                                                    converter_path)
    module = _load_module(converter_path)

    config = None
    if args.config_from_repo:
        from transformers import Sam3VideoConfig

        config = Sam3VideoConfig.from_pretrained(args.repo)
        print(f"[convert] config from {args.repo}/config.json "
              f"(image_size={getattr(config, 'image_size', '?')})")

    # Intercept the one call whose return value the converter throws away.
    captured: list[tuple[list, list]] = []
    original_load = torch.nn.Module.load_state_dict

    def recording_load(self, state_dict, *a, **kw):
        result = original_load(self, state_dict, *a, **kw)
        captured.append((list(result.missing_keys), list(result.unexpected_keys)))
        return result

    torch.nn.Module.load_state_dict = recording_load
    try:
        module.convert_sam3_checkpoint(checkpoint_path=checkpoint, output_path=args.out,
                                       config=config, push_to_hub=False, repo_id=None)
    finally:
        torch.nn.Module.load_state_dict = original_load

    if args.sync_processor:
        _sync_repo_processor(os.path.dirname(checkpoint), args.out)

    if not captured:
        raise SystemExit("[convert] the converter never called load_state_dict — its "
                         "structure changed; re-read it before trusting any output.")
    # The top-level Sam3VideoModel load is last; submodule loads come first.
    missing, unexpected = captured[-1]
    code = _audit(missing, unexpected, args.strict_unexpected)

    print(f"\n[convert] converter commit: {sha}")
    print(f"[convert] output: {args.out}")
    if code == 0:
        print("[convert] next: point sam3.model_id at it and A/B against facebook/sam3:\n"
              f"            just sam-profile /data/bags/_frames <cfg-with-model_id>")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
