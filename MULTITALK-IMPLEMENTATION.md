# MeiGen-MultiTalk — Implementation Plan

**Repo to create:** `storystudio-multitalk` (new, separate from both existing Wan2.2 workers)
**Endpoint target:** New RunPod serverless endpoint
**GPU target:** RTX 6000 Ada (47.4 GB)
**Date started:** 2026-08-05
**Model card:** https://huggingface.co/MeiGen-AI/MeiGen-MultiTalk
**Code repo:** https://github.com/MeiGen-AI/MultiTalk (NeurIPS 2025)
**Paper:** arXiv 2505.22647
**License:** Apache 2.0

---

## Update 2026-08-05 — InfiniteTalk changes the recommendation

Checked InfiniteTalk's actual HF repo file tree
(`huggingface.co/MeiGen-AI/InfiniteTalk/tree/main/quant_models`) rather than
relying on the README summary. It ships more than a single FP8 file — it's
a small matrix of pre-quantized, single-file DiT checkpoints:

| File | Size | Notes |
|---|---|---|
| `infinitetalk_single_fp8.safetensors` | 19.5 GB | Single-person, FP8, standard 40-step sampling |
| `infinitetalk_single_fp8_lora.safetensors` | 19.5 GB | Single-person, FP8, **byte-identical size to the non-lora file** |
| `infinitetalk_multi_fp8.safetensors` | 19.5 GB | Multi-person, FP8 |
| `infinitetalk_multi_fp8_lora.safetensors` | 19.5 GB | Multi-person, FP8 + LoRA variant |
| `infinitetalk_single_int8.safetensors` / `_lora` | ~19.5 GB each | INT8 equivalents |
| `infinitetalk_multi_int8.safetensors` / `_lora` | ~19.5 GB each | INT8 equivalents |
| `t5_fp8.safetensors` | 6.7 GB | Quantized T5 text encoder |

**Why this matters:** each `_lora` file is the exact same tensor
shapes/byte count as its non-`_lora` counterpart — strong evidence the
lightx2v 4-step LoRA has been **merged into the weights before
quantization**, not shipped as a separate delta file. If confirmed at
inference time, this means MeiGen is already publishing the equivalent of
what we built by hand for Wan2.2 (`lightx2v/Wan2.2-Distill-Models`'
single-file, pre-merged, native-4-step FP8 checkpoint) — for InfiniteTalk,
officially, out of the box. That directly resolves the "Phase 2 / stretch
goal" FP8 question this doc originally left open for MultiTalk, and does it
without us needing to design a custom loader that stitches a quantized base
+ an unquantized audio patch + a separately-applied LoRA together.

**This changes the recommendation: build on InfiniteTalk, not MultiTalk,
as the primary target.** InfiniteTalk is MeiGen's own successor project —
same Wan2.1-I2V-14B-480P lineage (so the base-model-mismatch conclusion
below still holds unchanged relative to `wan22-14B-fp8-4step`), but with
better documented lip sync, less hand/body distortion (MeiGen's own claim),
longer max duration (1000 frames / ~40s vs MultiTalk's 15s), video-to-video
mode, and now an officially-shipped single-file FP8(+LoRA) checkpoint that
looks like a much closer match to our existing "one quantized safetensors
file, native few-step sampling" operating model than anything documented
for MultiTalk itself.

**VRAM implication:** a 19.5 GB single-file FP8 DiT is *smaller* than the
~30 GB (2×15 GB) both-DiTs-resident footprint of the existing Wan2.2
Lightning worker on the same 47.4 GB card. Even adding the FP8 T5 (6.7 GB),
wav2vec (~1 GB), and VAE (~0.5 GB), resident weights land around ~27 GB —
leaving meaningfully more headroom than the current worker for
`--num_persistent_param_in_dit` to be raised well above the `0`
(CPU-paged) low-VRAM baseline, i.e. likely faster steady-state generation
on this card than InfiniteTalk's own documented 24 GB/RTX 4090 config.

**What's still unverified** (worth resolving in Phase 3 below before
committing to this path):
- The *exact* CLI to invoke the `_lora` quant files. The only documented
  quantized-inference example in the README uses the plain
  `infinitetalk_single_fp8.safetensors` at `--sample_steps 40` — the
  `_lora` variants exist in the repo but aren't walked through in the
  README text. Confirm whether `--quant_dir <..._lora.safetensors>` alone
  (at `--sample_steps 4`) is sufficient, or whether `--lora_dir` must
  *also* be passed (which would contradict the "already merged" theory —
  worth testing both ways).
- Whether the FP8 quant path composes with `--use_teacache` / `--use_apg`.
- `t5_fp8.safetensors`'s exact loading flag (not confirmed from the README
  fetch — likely auto-detected from `--quant_dir`'s sibling files, or a
  separate `--t5_quant` style flag; check `generate_infinitetalk.py`
  directly).
- Quality delta of FP8+LoRA (4-step, quantized) vs the bf16/40-step
  baseline — same kind of validation we did for Wan2.2 Lightning's
  motion-artifact check.

The rest of this document was written against MultiTalk before this check;
sections below still apply to InfiniteTalk with the following renames:
`generate_multitalk.py` → `generate_infinitetalk.py`, `multitalk.safetensors`
→ `infinitetalk.safetensors` (or the `single/multi` + `quant_models/`
variants above), `MeiGen-AI/MeiGen-MultiTalk` → `MeiGen-AI/InfiniteTalk`.
The base-model download (Wan2.1-I2V-14B-480P), audio encoder
(chinese-wav2vec2-base), and the core "why not `wan22-14B-fp8-4step`"
reasoning are unchanged and apply identically to InfiniteTalk.

---

## Update 2026-08-06 — corrected `quant_models` file listing (single has no `_lora` fp8 file)

Checked the live HF repo listing directly
(`huggingface.co/api/models/MeiGen-AI/InfiniteTalk`) rather than relying on
the file table in the "Update 2026-08-05" section above, which turns out to
be **wrong on one point**: `infinitetalk_single_fp8_lora.safetensors` does
**not exist** in the repo. The actual `quant_models/` contents are:

```
infinitetalk_multi_fp8.safetensors        + infinitetalk_multi_fp8.json
infinitetalk_multi_fp8_lora.safetensors   + infinitetalk_multi_fp8_lora.json
infinitetalk_multi_int8.safetensors       + infinitetalk_multi_int8.json
infinitetalk_multi_int8_lora.safetensors  + infinitetalk_multi_int8_lora.json
infinitetalk_single_fp8.safetensors       + infinitetalk_single_fp8.json
infinitetalk_single_int8.safetensors      + infinitetalk_single_int8.json
infinitetalk_single_int8_lora.safetensors + infinitetalk_single_int8_lora.json
t5_fp8.safetensors + t5_map_fp8.json
quant.json
```

Two things this changes:
1. **Only `multi` has a `_lora` fp8 file.** `single`-person jobs can't use
   the "official 4-step FP8+LoRA" path at all — they fall back to plain
   `fp8` (40-step default guidance/steps) or the `int8_lora` file (which
   does exist for `single`) for a fast path. This also weakens the
   "already-merged" theory from the 2026-08-05 update, since there was
   never a same-sized `_lora`/non-`_lora` pair to compare for `single` in
   the first place — worth re-verifying the merge theory against the
   `multi` pair specifically once real inference is run.
2. **Every quant `.safetensors` file ships a same-named sidecar `.json`**
   (scale/mapping metadata, e.g. `infinitetalk_multi_fp8.json`) not
   mentioned in the original size table — these need to be downloaded and
   loaded alongside their `.safetensors` counterpart, not just the big file.

`format.json.example` and `model_server.py`'s fallback config in
`storystudio-multitalk` have been corrected to match (per-variant
`default_weight_format`: `single` → `fp8`, `multi` → `fp8_lora`). Loading
the sidecar `.json` files during DiT swap is still a TODO in
`_load_dit_state_dict()`.

---

## What This Is

Audio-driven, multi-person conversational video generation. Given a
reference image (one or more people) plus one WAV file per person, MultiTalk
animates lip sync + head/body motion for each speaker simultaneously,
supporting singing, cartoon characters, and up to 15s clips at 480p/720p.

It is **not a standalone model** — it's a set of additional weights
(`multitalk.safetensors`) that patch audio cross-attention modules into an
existing text-to-video/image-to-video DiT, plus a Label Rotary Position
Embedding (L-RoPE) scheme that binds each person's audio stream to their
region of the frame.

---

## Requirement: both single-person and two-person modes

Confirmed requirement: the worker needs to serve **both** one person
talking and two people talking (conversation/duet), not just one mode.
This has a concrete design implication that MultiTalk's docs don't
surface, because InfiniteTalk (our recommended target, see above) ships
**separate checkpoint files** for the two cases rather than one
unified model:

- `single/infinitetalk.safetensors` (bf16) / `quant_models/infinitetalk_single_fp8*.safetensors`
- `multi/infinitetalk.safetensors` (bf16) / `quant_models/infinitetalk_multi_fp8*.safetensors`

(MultiTalk, by contrast, has just one `multitalk.safetensors` used for both
`single_example_1.json` and the 2-person `multitalk_example_2.json` — so
this single-vs-multi checkpoint split is InfiniteTalk-specific, presumably
because MeiGen fine-tuned separate weights per person-count for quality.)

**VRAM math for serving both:**

| Approach | Resident VRAM (FP8+LoRA DiT only) | Headroom left on 47.4 GB | Verdict |
|---|---|---|---|
| Keep only `single` resident | 19.5 GB | ~27 GB minus T5/wav2vec/VAE | Fine for 1-person jobs; 2-person jobs need a reload |
| Keep only `multi` resident | 19.5 GB | same | Fine *if* multi checkpoint also handles 1-person well (unverified) |
| Keep **both** resident simultaneously | 19.5 + 19.5 = 39 GB | ~7 GB minus T5/wav2vec/VAE (~1.5 GB) → **~5.5 GB for activations** | Same ballpark as the Wan2.2 worker's tightest OOM-adjacent config — risky, especially past 480p or short clips. Not recommended as the default. |

Recommended design, in priority order:
1. **Test first whether the `multi` checkpoint alone can serve 1-person
   requests** (just populate `cond_audio.person1`, omit `person2`) at
   acceptable quality vs the dedicated `single` checkpoint. If yes, this
   collapses the whole problem — one resident checkpoint, zero swap logic,
   ~27 GB resident, comfortable headroom. This should be the first thing
   checked in Phase 3, before any handler work.
2. If quality genuinely requires the dedicated `single` checkpoint for
   solo jobs, swap checkpoints on demand: T5/wav2vec/VAE/wav2vec stay
   warm, only the ~19.5 GB DiT safetensors file gets reloaded from the
   network volume when a job's person-count differs from whatever's
   currently loaded. This is a `load_state_dict`-style swap (seconds,
   volume-throughput-bound), not a cold start (~170-190s) — same
   "persistent model server, socket-driven" pattern as the existing worker,
   extended with a "which DiT is currently loaded" check before dispatch.
3. Only if traffic patterns justify the extra cost: run **two pinned
   endpoints** (one `single`-only, one `multi`-only), same tradeoff
   already documented for Active Workers in `API_DOCUMENTATION.md` — avoids
   swap latency entirely at the cost of double the always-on GPU spend.

Start with option 1 — it's a single test run, and if it holds it avoids
building swap logic at all.

---

## Answering the key question: can we reuse `wan22-14B-fp8-4step`?

**No, not directly.** The two projects sit on different base-model
generations:

| | `wan22-14B-fp8-4steps` (existing) | MultiTalk |
|---|---|---|
| Base model | **Wan2.2** I2V-A14B — MoE, dual DiT (high-noise + low-noise, 40 layers each, boundary-switched) | **Wan2.1**-I2V-14B-480P — single DiT, no MoE split |
| Distillation | `lightx2v/Wan2.2-Distill-Models` — full-weight replacement, native 4-step, FP8 e4m3 with per-channel scale | MultiTalk itself is full-precision (bf16); a *separate* `lightx2v` 4-step **LoRA** exists for Wan2.1-I2V-14B-480P, applied on top, not baked in |
| Audio conditioning | none | `multitalk.safetensors` adds new cross-attention sub-modules per transformer block + L-RoPE — these keys don't exist in Wan2.2's architecture at all |

MeiGen has not published (and as far as the current research shows, nobody
in the community has published) a MultiTalk checkpoint retrained against
Wan2.2's MoE architecture. `multitalk.safetensors` was trained against, and
its `diffusion_pytorch_model.safetensors.index.json` patch specifically
targets, Wan2.1-I2V-14B-480P's block layout. Dropping it onto our Wan2.2
dual-DiT weights would be a key-name mismatch, not a supported
configuration — this would require MeiGen (or us) to retrain the audio
cross-attention weights against Wan2.2, which is out of scope here.

**What *is* reusable** is the infrastructure pattern we already built, not
the weights:
- RunPod serverless scaffold: `handler.py` (thin, validates input, uploads
  to R2) → persistent socket-based `model_server.py` (loads model once,
  stays warm) — see `wan22-14B-fp8-4steps/handler_v2.py` /
  `model_server.py`.
- `FP8Linear` custom quantization wrapper (weight stays FP8 on GPU, dequant
  on forward, optional dequant cache) — same technique could, in principle,
  be pointed at Wan2.1's DiT weights later (see **Phase 2 — Optional FP8
  path**, below). Not required for a first working version.
- R2 upload / asset naming convention (`storystudio/{category}/...`).
- Dockerfile structure: clone upstream repo at build time, pin
  torch/torchvision to a cu128 wheel that runs on both Ada and Blackwell,
  best-effort FlashAttention2 install with SDPA fallback.
- `format.json`-style dispatch if we later add multiple weight formats.

So: **new repo, new weights, same operational skeleton.**

---

## Required Model Downloads

| Component | HF repo | Purpose | Approx size |
|---|---|---|---|
| Wan2.1-I2V-14B-480P | `Wan-AI/Wan2.1-I2V-14B-480P` | Base I2V DiT + VAE + T5 | ~28.6 GB (bf16 DiT) + T5/VAE |
| MeiGen-MultiTalk | `MeiGen-AI/MeiGen-MultiTalk` | `multitalk.safetensors` (audio cross-attn weights) + patched `diffusion_pytorch_model.safetensors.index.json` | small (~few GB) |
| chinese-wav2vec2-base | `TencentGameMate/chinese-wav2vec2-base` | Audio encoder (works for English audio too, despite the name) | ~1.2 GB |
| Kokoro-82M (optional) | `hexgrad/Kokoro-82M` | TTS, only needed for `--audio_mode tts` (text-to-speech-driven avatars instead of pre-recorded WAV) | ~330 MB |
| lightx2v 4-step LoRA (optional, for speed) | `Kijai/WanVideo_comfy` → `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors`, or the dedicated I2V LoRA collection at `wangkanai/wan21-lightx2v-i2v-14b-480p` | Step-distillation LoRA, applied via `--lora_dir`, drops sampling to 4 steps | ~300 MB–2.8 GB depending on rank |

```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download MeiGen-AI/MeiGen-MultiTalk --local-dir ./weights/MeiGen-MultiTalk
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
```

### Directory layout (per upstream README)

```
weights/
├── Wan2.1-I2V-14B-480P/
│   ├── diffusion_pytorch_model.safetensors.index.json   ← REPLACED by MeiGen-MultiTalk's version
│   └── multitalk.safetensors                            ← copied/symlinked in from MeiGen-MultiTalk
├── chinese-wav2vec2-base/
├── Kokoro-82M/                (optional)
└── MeiGen-MultiTalk/
```

On our network-volume setup this maps to something like
`/runpod-volume/multitalk/{Wan2.1-I2V-14B-480P,MeiGen-MultiTalk,chinese-wav2vec2-base}/`,
mirroring the `wan22-lightning-fp8` volume layout already in use.

---

## VRAM Budget (RTX 6000 Ada, 47.4 GB)

Upstream numbers (not yet independently verified on our hardware):

| Mode | GPU | VRAM | Notes |
|---|---|---|---|
| Low-VRAM (`--num_persistent_param_in_dit 0`) | RTX 4090 | ~24 GB | Params paged CPU↔GPU per-forward; slower but fits |
| Default (some params resident) | A100 | reported ~80 GB in places, unverified | Likely full bf16 DiT + T5 + activations resident, no offload |
| Multi-GPU (`--ulysses_size 8`, FSDP) | 8×A100 | N/A | For scaling multi-person / long clips, not needed at our scale |

Our card (47.4 GB) sits comfortably between the two single-GPU modes: it's
2× the confirmed-working 24 GB low-VRAM configuration, so a first pass
should just use `--num_persistent_param_in_dit 0` (known-good, matches our
existing card class) and then experiment with raising it for speed once
correctness is confirmed — same empirical-tuning approach we used for the
Wan2.2 Lightning worker's `DEQUANT_CACHE_RESERVE_GB`.

Rough resident-weight budget at bf16:
- Wan2.1-I2V-14B DiT: ~28.6 GB (bf16, single model — no dual-DiT doubling like Wan2.2's MoE)
- T5-XXL: same as our existing workers, offload to CPU except during encode
- chinese-wav2vec2-base: small, <1 GB
- VAE: ~0.5 GB
- multitalk.safetensors (audio cross-attn add-on): small, a few hundred MB

This leaves meaningfully more headroom than the Wan2.2 Lightning worker
(which resident-loads *two* 15 GB FP8 DiTs simultaneously = ~30 GB just for
weights). Even at full bf16 with no quantization, one Wan2.1 DiT should fit
with room for activations at 480p; 720p and longer clips (10-15s) are where
we'll need to lean on `--num_persistent_param_in_dit` tuning or INT8
quantization.

---

## Acceleration: 4-step LoRA (recommended for production latency)

Full default sampling is **40 steps**, guidance scale 5.0 (text) / 4.0
(audio) — this is the multi-minute-per-clip regime we specifically moved
away from with the Wan2.2 Lightning worker. MultiTalk supports the same
kind of step-distillation via LoRA rather than full weight replacement:

```bash
python generate_multitalk.py \
  --ckpt_dir weights/Wan2.1-I2V-14B-480P \
  --wav2vec_dir weights/chinese-wav2vec2-base \
  --lora_dir weights/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \
  --input_json examples/single_example_1.json \
  --sample_steps 4 \
  --sample_text_guide_scale 1.0 \
  --sample_audio_guide_scale 2.0 \
  --mode streaming \
  --num_persistent_param_in_dit 0 \
  --use_teacache
```

Key parameter deltas when the LoRA is active (mirrors what we learned
tuning the Wan2.2 Lightning worker — CFG-distilled models need scale/step
changes, not just fewer steps):

| Parameter | Default (40-step) | With lightx2v LoRA (4-step) |
|---|---|---|
| `sample_steps` | 40 | 4 |
| Text guidance scale | 5.0 | 1.0 |
| Audio guidance scale | 4.0 | 2.0 |

An 8-step alternative (`vrgamedevgirl84/Wan14BT2VFusioniX`,
`FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors`) is also documented
upstream as a quality/speed middle ground, same pattern as our own
"6-8 step" fallback recommendation for the Wan2.2 worker when 4-step
artifacts show up on hard motion.

`--use_teacache` (2-3× speedup, `--teacache_thresh` 0.2-0.5) and `--use_apg`
(color-tone consistency across streaming segments) are additional
upstream-supported flags worth testing once basic inference is validated.

---

## Quantization

MultiTalk ships **INT8** support natively, not FP8:

```bash
python generate_multitalk.py \
  --quant int8 \
  --quant_dir weights/MeiGen-MultiTalk \
  ... (same as above)
```

This needs an extra quantized-weights file
(`weights/MeiGen-MultiTalk/quant_models/quant_model_int8_FusionX.safetensors`
in the upstream examples — check the actual MeiGen-MultiTalk HF repo tree
for the exact filename/pairing with whichever LoRA is active) and is
documented as single-GPU only.

**No native FP8 path exists in the MultiTalk repo today.** The broader
lightx2v ecosystem does publish FP8 *full-weight* distilled Wan2.1-I2V-14B
checkpoints (e.g. `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v`),
analogous to what we used for Wan2.2 — but those are alternate base weights,
and `multitalk.safetensors`'s patched `index.json` assumes the *stock*
Wan2.1-I2V-14B-480P shard layout. Swapping in a differently-sharded/quantized
base without adapting the patch is unverified and would need the same kind
of key-mapping investigation we did for the Wan2.2 Lightning loader (see
`WAN22-I2V-LIGHTNING-4STEP-BUILD.md`'s "New Model Loader" section for the
template of that investigation). Treat this as a **Phase 2 / stretch goal**,
not a blocker for a working v1: INT8 + the 4-step LoRA should already get us
into a similar ballpark of speed to the Wan2.2 Lightning worker, on a card
with more headroom to spare.

---

## API Design (mirrors `handler_v2.py` conventions)

Unlike the I2V workers, MultiTalk needs **per-person audio**, not just an
image + text prompt. Upstream's own request shape (`input_json`):

```json
{
  "prompt": "A woman is passionately singing into a professional microphone...",
  "cond_image": "<url or path>",
  "cond_audio": {
    "person1": "<url or path to wav>",
    "person2": "<url or path to wav, omit for single-person>"
  },
  "audio_type": "para"
}
```

`audio_type` controls multi-person timing: `"para"` = simultaneous
(singing together), likely `"add"`/sequential for turn-taking conversation
(verify exact enum against `generate_multitalk.py` — not confirmed from the
README alone, check the repo's `utils/` or `kokoro_utils.py` before wiring
this into the handler).

Proposed StoryStudio-facing payload, keeping the same shape family as the
I2V workers' `image`/`prompt`/`resolution`/`duration_s`:

```json
{
  "input": {
    "image": "https://.../scene.png",
    "prompt": "A woman passionately singing into a microphone",
    "audio": {
      "person1": "https://.../speaker1.wav",
      "person2": "https://.../speaker2.wav"
    },
    "audio_type": "para",
    "resolution": "480p",
    "sample_steps": 4,
    "duration_s": 5
  }
}
```

Handler responsibilities (new vs. the I2V workers):
- Download/base64-decode **N audio files**, not just one image — extend
  `verify_model_present()` / job-input validation accordingly.
- Route each job to the right checkpoint based on `len(audio.keys())` — 1
  key → single-person, 2 keys → multi-person — per the **Requirement: both
  single-person and two-person modes** section above. If Phase 3 confirms
  the `multi` checkpoint alone covers both cases, this routing collapses
  to a no-op; otherwise the model server needs a "swap DiT if person-count
  changed" check before dispatch.
- No `n_prompt` concept carries over cleanly; check whether MultiTalk uses
  it at all with the LoRA active (same "CFG disabled with distillation"
  situation we hit with Wan2.2 Lightning).
- Duration cap: upstream claims up to 15s (201 frames @ 25fps) but flags 81
  frames as "optimal for prompt adherence" — expect to empirically
  re-derive a safe `frame_num` ceiling for 47.4 GB the same way we did for
  the Wan2.2 worker (113-frame cap found via OOM testing, not from docs).

---

## Build Plan — Step by Step

### Phase 1 — Repo Setup
- [ ] Create GitHub repo `storystudio-multitalk`
- [ ] Copy Dockerfile skeleton from `storystudio-wan2-i2v-lightning` (cu128 torch pin, best-effort FlashAttention2, R2/boto3/runpod deps) — swap the `git clone` target to `MeiGen-AI/MultiTalk`
- [ ] `MODEL_PATH` → `/runpod-volume/multitalk`

### Phase 2 — Model Download
- [ ] Download Wan2.1-I2V-14B-480P, MeiGen-MultiTalk, chinese-wav2vec2-base to the network volume (CPU/storage pod, verify `/runpod-volume` mount as in the existing gotcha note)
- [ ] Apply the `index.json` replacement + `multitalk.safetensors` link per upstream layout
- [ ] Download the lightx2v 4-step LoRA for Wan2.1-I2V-14B-480P

### Phase 3 — Local/Pod Functional Test (bf16, no quant, default steps)
- [ ] Run `generate_multitalk.py`/`generate_infinitetalk.py` directly on a rented pod with the single-person example to confirm the base pipeline works before any of our modifications
- [ ] Confirm single-person lip sync quality baseline
- [ ] **Decisive test:** run the `multi` checkpoint with only `person1` populated (no `person2`) and compare quality against the dedicated `single` checkpoint on the same input — determines whether we need checkpoint-swap logic at all (see **Requirement: both single-person and two-person modes**)
- [ ] Confirm 2-person `multi` checkpoint quality on a real conversation (not same-audio-for-both-speakers like the upstream demo)

### Phase 4 — Speed Path
- [ ] Re-run with `--lora_dir` (4-step) + adjusted guidance scales, confirm quality is acceptable
- [ ] Test `--num_persistent_param_in_dit 0` vs raising it, measure VRAM/speed tradeoff on the 47.4 GB card
- [ ] Test `--use_teacache`

### Phase 5 — Handler/Model-Server Port
- [ ] Adapt `model_server.py`'s persistent-process + Unix-socket pattern to wrap `generate_multitalk.py`'s pipeline instead of `WanI2V`
- [ ] Adapt `handler_v2.py` for multi-audio-file input, R2 fetch/upload, new `storystudio/video/multitalk_*` asset naming
- [ ] Multi-person validation (2-speaker `audio_type: para` test)

### Phase 6 — Docker Build & Deploy
- [ ] Build image, push via the same GH Actions pattern as the existing workers
- [ ] Deploy to new RunPod endpoint, attach network volume
- [ ] Empirically find the safe `frame_num`/duration ceiling for 47.4 GB (same OOM-boundary-testing approach as the Wan2.2 worker's 113-frame cap)

### Phase 7 — Quality Validation
- [ ] Single-person lip sync quality vs known-good references
- [ ] Multi-person timing/binding correctness (L-RoPE person-to-region binding)
- [ ] INT8 quantization quality delta, if used in production

---

## Open Questions

1. **Exact `audio_type` enum values** — only `"para"` confirmed from the example JSON; need to check `generate_multitalk.py`/`utils/` for the sequential/turn-taking option name.
2. **`quant_model_int8_*.safetensors` pairing** — does the INT8 quant model need to match whichever LoRA (FusioniX vs lightx2v) is active, or is it LoRA-agnostic? Upstream example ties it to FusioniX specifically.
3. **720p on a single GPU** — upstream README describes 720p as multi-GPU-only ("update forthcoming" for single-GPU 720p at time of writing); confirm current status before promising 720p in our API.
4. **Person localization input** — the example JSON has no explicit bounding-box field, suggesting fully automatic face/region binding from `cond_audio` person ordering. Confirm this holds for arbitrary (non-example) images, especially crowded or ambiguous compositions.
5. **FP8 base-model swap (Phase 2/stretch)** — whether a lightx2v FP8 full-weight Wan2.1-I2V-14B checkpoint can host `multitalk.safetensors`'s patch without a custom loader, following the same investigation pattern as `WAN22-I2V-LIGHTNING-4STEP-BUILD.md`.
6. **Does the `multi` checkpoint alone cover single-person jobs well?** This is the highest-priority open question for the two-person requirement — see **Requirement: both single-person and two-person modes**. Resolves whether we need any checkpoint-swap logic at all.
7. **`multi` checkpoint's max speaker count** — confirmed to support 2 in upstream examples; unclear if 3+ works or if it's architecturally capped at 2 via L-RoPE's slot design.

---

## Reference Links

- Model card: https://huggingface.co/MeiGen-AI/MeiGen-MultiTalk
- Code: https://github.com/MeiGen-AI/MultiTalk
- Paper: https://arxiv.org/abs/2505.22647
- Base model: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P
- Audio encoder: https://huggingface.co/TencentGameMate/chinese-wav2vec2-base
- TTS (optional): https://huggingface.co/hexgrad/Kokoro-82M
- 4-step LoRA (I2V-specific collection): https://huggingface.co/wangkanai/wan21-lightx2v-i2v-14b-480p
- 4-step LoRA (used in upstream MultiTalk examples): `Kijai/WanVideo_comfy` — `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors`
- 8-step LoRA alternative: `vrgamedevgirl84/Wan14BT2VFusioniX` — `FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors`
- Successor project (better lip sync, worth tracking): https://github.com/MeiGen-AI/InfiniteTalk (same Wan2.1-I2V-14B-480P base, ships its own `infinitetalk_single_fp8.safetensors`)
- Existing worker for pattern reference: `~/wan22-14B-fp8-4steps/` (`WAN22-I2V-LIGHTNING-4STEP-BUILD.md`, `handler_v2.py`, `model_server.py`, `Dockerfile`)
