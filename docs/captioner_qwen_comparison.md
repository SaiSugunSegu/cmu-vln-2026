# Captioner VLM comparison — Qwen2.5-VL vs Qwen3-VL

Benchmark of the captioner backends on ScanNet `scene0000_01` instance crops.
Goal: choose a default for robot deploy where the captioner shares a 24 GB GPU
with other models (e.g. SAM3).

**Date:** 2026-07-29
**Hardware:** NVIDIA A10G 23 GB (CUDA 12.2), measured in `iros2026_ai_module`
**Stack:** torch 2.5.1+cu121, transformers 5.14.1, bitsandbytes 0.50
**Workload:** 72 crops from
`SORT3D/data/captions/Scannet/scene0000_01/instance_crops`,
`max_new_tokens=200`, same prompt and vision-token budget for both backends
(`min_pixels=4×28×28`, `max_pixels=640×28×28`), untimed warmup before timing.

Outputs (captions + `timing.json`) live under:

```
SORT3D/data/captions/Scannet/scene0000_01/
  qwen2_5vl/          # int4
  qwen3vl/            # int4
  qwen2_5vl_bf16/
  qwen3vl_bf16/
  qwen2_5vl_awq/      # AWQ, measured in a disposable transformers 4.57 container
```

Reproduce with:

```bash
# inside iros2026_ai_module
python3 -m captioner.models.captioning <crops_dir> \
  --captioning_model {qwen2_5vl|qwen3vl} \
  --quantization {int4|none} \
  --batch_size 8 \
  --output_dir <out_dir>
```

---

## 1. Headline timings (batch size 8)

| Model | Quant | Total gen | s/crop | tok/s | Tokens | Load | Peak allocated |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | int4 (nf4) | 45.4 s | 0.63 | 82.3 | 3734 | 62 s | 2.9 GB |
| Qwen3-VL-4B | int4 (nf4) | 74.0 s | 1.03 | 72.9 | 5388 | 72 s | 3.6 GB |
| Qwen2.5-VL-3B | bf16 | 36.1 s | 0.50 | 102.8 | 3710 | 57 s | 7.5 GB |
| Qwen3-VL-4B | bf16 | 59.1 s | 0.82 | 77.8 | 4602 | 67 s | 9.0 GB |

Qwen2.5-3B is faster per token (~13% at int4, ~32% at bf16). Qwen3 is also more
verbose (~44% more tokens at int4), which further widens the wall-clock gap.
bitsandbytes int4 is *slower* than bf16 when the captioner has the GPU alone at
batch 8, because of dequantization overhead — but that reverses under a shared
GPU at larger batches (see §3).

---

## 2. Caption quality (sample)

| Crop | Qwen2.5-VL-3B int4 | Qwen3-VL-4B int4 |
|---|---|---|
| `1_scale` | silver, metallic, rectangular; mentions digital display | white/gray, plastic+metal; correctly identifies body scale |
| `19_trash-can` | black, plastic, cylindrical + lid/handle detail | black/dark brown, plastic, cylindrical (terse) |
| `26_guitar-case` | **black, made of wood** (wrong material/color) | **beige, fabric** (correct) |
| `27_couch` | dark blue, fabric; long multi-sentence description | dark blue, fabric, L-shaped (terse but correct shape) |

Qwen3 is more accurate on harder crops but sometimes stops after one clause.
Qwen2.5 reliably produces 3–4 descriptive sentences but can hallucinate material
and color. Quality is a real trade-off, not a wash.

---

## 3. Memory and batch-size sweep (shared-GPU view)

Measured on 16 crops. **Reserved** memory is what the CUDA caching allocator
holds and therefore what another model sharing the GPU cannot use.

| Config | Weights reserved | Peak reserved @ b16 | s/crop @ b1 | s/crop @ b8 | s/crop @ b16 | tok/s @ b16 |
|---|---|---|---|---|---|---|
| Qwen2.5 bf16 | 7.06 GB | 7.52 GB | 2.35 | 0.51 | **0.30** | 187 |
| Qwen2.5 int4 | 2.51 GB | 3.01 GB | 2.71 | 0.80 | **0.37** | 148 |
| Qwen3 bf16 | 8.27 GB | 9.35 GB | 2.70 | 0.96 | **0.52** | 114 |
| Qwen3 int4 | **2.95 GB** | **3.95 GB** | 3.61 | 1.00 | **0.53** | 112 |

Key observations:

- Memory is weight-dominated. Raising batch size from 1 → 16 adds only
  ~0.4–0.6 GB while roughly doubling throughput.
- At batch 16, Qwen3 int4 and bf16 are within measurement noise
  (0.53 vs 0.52 s/crop) while int4 saves **~5.4 GB**.
- For Qwen2.5, int4 still costs ~23% throughput vs bf16 at batch 16, but saves
  ~4.5 GB.

### Estimated captioner-node budget on a shared 4090

The captioner process also holds OpenCLIP `DFN5B-CLIP-ViT-H-14-378` (~2 GB bf16,
estimated — not measured here) plus ~0.3–0.5 GB CUDA context per process
(`nvidia-smi` will read higher than the torch stats above).

| Component | Approximate VRAM |
|---|---|
| Qwen3-VL-4B int4 @ batch 16 | ~4 GB peak reserved |
| OpenCLIP ViT-H (bf16) | ~2 GB |
| CUDA context | ~0.5 GB |
| **Captioner node total** | **~6–7 GB** |
| Remaining on 24 GB card | **~17 GB for SAM3 + rest** |

The 4090 (Ada, ~1008 GB/s) is more bandwidth-rich than the A10G (~600 GB/s).
Decode is memory-bandwidth-bound, so expect roughly **0.3 s/crop for Qwen3-int4**
and **~0.2 s/crop for Qwen2.5-int4** at batch 16 — extrapolation, not measured.

---

## 4. AWQ checkpoint (optional path)

`Qwen/Qwen2.5-VL-3B-Instruct-AWQ` was also timed, but **cannot load in the
deployed image** (transformers 5.x routes AWQ through `gptqmodel`, which needs
torch ≥ 2.8; the image has torch 2.5.1). Measured in a disposable container with
transformers 4.57.1 + autoawq 0.2.9 + `autoawq_kernels`:

| Config | Total gen | s/crop | tok/s | Tokens | Load | Peak allocated |
|---|---|---|---|---|---|---|
| AWQ (official) | 39.9 s | 0.55 | 77.6 | 3096 | **24.3 s** | 3.7 GB |
| bitsandbytes nf4 | 42.9 s | 0.60 | 85.2 | 3660 | 62.2 s | 3.5 GB |
| bf16 | 35.3 s | 0.49 | 102.8 | 3626 | 53.5 s | 7.5 GB |

AWQ's real win is **cold start** (24 s vs 62 s), not generation speed. Per-token
it was the slowest of the three. AutoAWQ is deprecated/unmaintained; prefer a
pre-quantized bitsandbytes checkpoint (`save_pretrained` after one-time
quantize) if cold start becomes the bottleneck.

---

## 5. Recommendation for 4090 + SAM3

**Keep the current deploy defaults: `qwen3vl` + `int4` + `batch_size=16`.**

| Concern | Choice | Why |
|---|---|---|
| Shared 24 GB with SAM3 | int4, not bf16 | Same speed at batch 16, ~5 GB back for SAM3 |
| Throughput | batch 16 | Free win vs batch 8; activations barely grow |
| Quality vs Qwen2.5 | stay on Qwen3 | Better material/color accuracy on hard crops |
| Cold start (60–72 s) | optional follow-up | Export a pre-quantized 4-bit local checkpoint |

Deploy knobs (already set this way):

- `captioning_node.py` / `sort3d_gt_semantics_launch.xml`: `captioning_model=qwen3vl`, `batch_size=16`
- `Captioner`: `captioning_quantization="int4"`

Do **not** chase AWQ for this deploy unless cold start becomes the limiting
factor — the dependency path is fragile on the current stack.

---

## 6. Code notes

Both backends share `QwenVLHFBackend` in
`ai_module/src/captioner/captioner/models/captioning.py` so prompt, pixel
budget, and decode path are identical. Defaults:

- Qwen2.5: `Qwen/Qwen2.5-VL-3B-Instruct` (base + bitsandbytes; the old
  `*-AWQ` default cannot load without autoawq/gptqmodel)
- Qwen3: `Qwen/Qwen3-VL-4B-Instruct`

The CLI writes `timing.json` next to captions when `--output_dir` is set.
Pre-quantized (`awq`/`gptq` in the model id) checkpoints skip bitsandbytes and
load in fp16 (AWQ kernels are fp16-only).
