# InfiniteTalk (RunPod Serverless)

Audio-driven, talking-avatar video generation. Given a reference image plus
one or two WAV files (one per speaker), animates lip sync + head/body motion
for each person simultaneously. Built on
[MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) — the
successor to MeiGen's MultiTalk, same Wan2.1-I2V-14B-480P base, with better
documented lip sync, longer max duration, and (unlike MultiTalk) officially
published single-file FP8/INT8 quantized checkpoints.

See [`MULTITALK-IMPLEMENTATION.md`](./MULTITALK-IMPLEMENTATION.md) for the
full build plan, VRAM budget, open questions, and the "Update 2026-08-05"
section explaining why this worker targets InfiniteTalk rather than
MultiTalk.

**Status: Phase 1 (repo scaffold) + Phase 5 (handler/model-server port)
only.** This repo has NOT been built, deployed, or tested against a real GPU
— see "Known gaps / what Phase 3 must verify" below before treating any of
this as working code.

## How this differs from the Wan2.2 I2V workers

| Aspect | `wan22-14B-fp8-4steps` | This worker |
|---|---|---|
| Base model | Wan2.2 I2V-A14B (MoE, dual DiT) | Wan2.1-I2V-14B-480P (single DiT) |
| Input | image + text prompt | image + text prompt + 1-2 audio tracks |
| Conditioning | none beyond text | audio cross-attention + L-RoPE person binding |
| Checkpoints | high/low-noise DiT (always both resident) | single-person / multi-person DiT (see below — only one resident at a time by default) |
| Quantization | custom `FP8Linear` wrapper (`quantize_wan22_fp8.py`) | pre-quantized single-file checkpoints shipped by MeiGen (`quant_models/infinitetalk_*_fp8*.safetensors`) — no custom wrapper needed/ported |

## Single vs. multi-person: checkpoint-swap design

InfiniteTalk ships **separate DiT checkpoint files** for single-person vs.
two-person (multi) generation, unlike MultiTalk's one unified model. This
worker implements "option 2" from the plan doc's *Requirement: both
single-person and two-person modes* section:

- T5 text encoder, wav2vec2 audio encoder, and the VAE stay **resident** in
  GPU memory for the life of the container.
- Only the ~19.5 GB DiT safetensors file gets **swapped** (a
  `load_state_dict`-style reload, not a cold restart) when a job's
  person-count differs from whatever variant is currently loaded. See
  `model_server.py`'s `ModelServer.load_dit()`.
- Routing: `handler_v2.py` sends 1 person → `single` checkpoint, 2 persons →
  `multi` checkpoint (`len(audio.keys())`).

**Option 1 (not implemented, higher priority to try first):** the plan doc
flags an open question — whether the `multi` checkpoint alone, given only
`person1`, already matches the dedicated `single` checkpoint's quality. If
Phase 3 GPU testing confirms this, the swap logic above becomes unnecessary
(`multi` loads once at startup and never swaps). See the `TODO` in
`model_server.py`'s module docstring.

## API Usage

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {api_key}" \
  -d '{
    "input": {
      "image": "https://example.com/scene.png",
      "prompt": "A woman passionately singing into a microphone",
      "audio": {
        "person1": "https://example.com/speaker1.wav"
      },
      "audio_type": "para",
      "resolution": "480p",
      "sample_steps": 4,
      "duration_s": 5
    }
  }'
```

For a two-person (multi) job, add `audio.person2`:

```json
"audio": {
  "person1": "https://example.com/speaker1.wav",
  "person2": "https://example.com/speaker2.wav"
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | required | - | Image URL or base64 encoded |
| `prompt` | string | optional | "" | Motion/action description |
| `audio` | object | required | - | `{"person1": <url or base64>, "person2": <optional>}` — 1 key routes to the `single` checkpoint, 2 keys route to `multi` |
| `audio_type` | string | optional | `"para"` | Multi-person timing. Only `"para"` (simultaneous) is confirmed against upstream's example — other values are passed through unchanged with a warning, not rejected, since the full enum isn't confirmed (plan doc Open Question 1) |
| `resolution` | string | optional | `"480p"` | `"480p"` or `"720p"`. 720p single-GPU support is unconfirmed (plan doc Open Question 3) |
| `sample_steps` | int | optional | weight-format default (4 for `fp8_lora`, 40 for `bf16`/`fp8`/`int8`) | 1-50 |
| `duration_s` | int | optional | - | 1-5s (see below); takes priority over `frame_num` if both given |
| `frame_num` | int | optional | 81 | Direct frame-count control, used when `duration_s` isn't given |
| `project_id` / `frame_id` | string | optional | - | StoryStudio asset-naming fields (see R2 key convention below) |

**`duration_s` range is conservative and unvalidated.** InfiniteTalk's own
documented duration math (in `MULTITALK-IMPLEMENTATION.md`) is internally
inconsistent (it quotes "15s = 201 frames @ 25fps", which doesn't reconcile
arithmetically). `handler_v2.py` uses `frame_num = duration_s*25 + 1` as a
best-guess formula (mirroring the `wan22-14B-fp8-4steps` worker's own
convention) over a deliberately narrow 1-5s range. The real safe ceiling for
this card needs the same kind of empirical OOM-boundary testing that found
`wan22-14B-fp8-4steps`' 113-frame/7s cap (plan doc Phase 6) — not yet done
here.

**Your audio must be LONGER than the requested `duration_s`, not just as
long.** Confirmed against `wan/multitalk.py`'s real `generate_infinitetalk()`
(line ~478): it silently drops an audio embedding if its length isn't
*strictly greater than* `frame_num`, then asserts every speaker survived
that filter — fails with `"Aduio file not exists or length not satisfies
frame nums."` (real upstream typo) otherwise. `model_server.py` now clamps
`frame_num` down to fit whatever audio was actually provided rather than
crashing, so a too-short clip degrades to a shorter output video instead of
a hard failure — but for `duration_s` to be honored as requested, give it
audio at least ~0.1-0.2s longer than that.

### Response

```json
{
  "video_url": "https://storyaistudio.app/storystudio/video/multitalk_20260806070512_abc123.mp4",
  "generation_time": 142.0,
  "video_size_mb": 2.5,
  "resolution": "480p",
  "sample_steps": 4,
  "frame_num": 81,
  "duration_s": 3.24,
  "variant": "single"
}
```

## Model Files (Network Volume)

Layout assumed under `MODEL_PATH` (default `/runpod-volume/multitalk`) — see
[`format.json.example`](./format.json.example):

```
/runpod-volume/multitalk/
├── Wan2.1-I2V-14B-480P/        # base T5 / VAE / tokenizer
├── chinese-wav2vec2-base/       # audio encoder (works for English too)
├── single/
│   └── infinitetalk.safetensors           # bf16, single-person
├── multi/
│   └── infinitetalk.safetensors           # bf16, multi-person
├── quant_models/
│   ├── infinitetalk_single_fp8.safetensors      # no _lora variant exists for single (verified)
│   ├── infinitetalk_single_fp8.json             # sidecar metadata, same-named per quant file
│   ├── infinitetalk_single_int8.safetensors
│   ├── infinitetalk_single_int8_lora.safetensors
│   ├── infinitetalk_multi_fp8.safetensors
│   ├── infinitetalk_multi_fp8_lora.safetensors
│   ├── infinitetalk_multi_int8.safetensors
│   ├── infinitetalk_multi_int8_lora.safetensors
│   ├── t5_fp8.safetensors
│   ├── t5_map_fp8.json
│   └── quant.json
└── format.json                  # copy from format.json.example
```

This layout is **verified 2026-08-06 against the live HF repo file listing**
(`huggingface.co/api/models/MeiGen-AI/InfiniteTalk`), via
`scripts/download_infinitetalk_checkpoints.py` — see that script for the
exact fetched paths. One correction vs the original plan doc's size table:
**`infinitetalk_single_fp8_lora.safetensors` does not exist upstream** —
only `multi` ships a `_lora` fp8 variant, so `single` jobs default to plain
`fp8` (40 steps) rather than the 4-step LoRA path (see
`MULTITALK-IMPLEMENTATION.md`'s "Update 2026-08-06" note and
`format.json.example`'s per-variant `default_weight_format`). Every quant
`.safetensors` file also ships a same-named sidecar `.json` (scale/mapping
metadata) that must be downloaded alongside it — `_load_dit_state_dict()` in
`model_server.py` doesn't consume it yet (TODO, flagged inline). Download
source: `https://huggingface.co/MeiGen-AI/InfiniteTalk` +
`https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P` +
`https://huggingface.co/TencentGameMate/chinese-wav2vec2-base` (all Apache
2.0 / permissive).

### Downloading

Two helper scripts (gitignored, not tracked in this repo — see `scripts/`,
same convention as `wan22-14B-fp8-4steps`) fetch everything above:

```
pip install -U huggingface_hub
python3 scripts/download_infinitetalk_checkpoints.py --dest /workspace/multitalk   # ~47 GB core quant set + format.json
python3 scripts/download_base_encoders.py            --dest /workspace/multitalk   # ~15 GB VAE/T5/CLIP + wav2vec2
```

`download_infinitetalk_checkpoints.py`'s default (no flags) fetches only the
"core" set actually used by `model_server.py`'s default per-variant weight
formats — `single`'s plain `fp8` + `multi`'s `fp8_lora`, plus the shared
`t5_fp8` files and each `.safetensors`' sidecar `.json`. Add
`--all-quant-variants` for the full int8/int8_lora + non-lora multi_fp8 set
(~+70 GB, only needed for Phase 7's quant-quality comparison), or
`--include-bf16` for the full-precision `single/`+`multi/` DiTs (~+28 GB,
needed for Phase 3's baseline correctness test). `download_base_encoders.py`
fetches CLIP by default (~2.6 GB, unverified whether InfiniteTalk actually
uses it — see the script's `CLIP_FILES` comment) and can skip it with
`--no-clip`.

**Mount-point gotcha:** run both on a RunPod CPU/storage Pod with the
network volume attached, and check `df -h` first — interactive Pods often
mount the network volume at `/workspace`, while `/runpod-volume` on that
same Pod is just local ephemeral disk that's wiped when the Pod is deleted.
Only the Serverless endpoint itself mounts the volume at `/runpod-volume`
(what `MODEL_PATH` above assumes).

## Docker Image

```
romantony/storystudio-multitalk:latest
```

Built and pushed to Docker Hub automatically by
[`.github/workflows/docker-build.yml`](./.github/workflows/docker-build.yml)
on every push to `main` (or manually via `workflow_dispatch`), mirroring
`wan22-14B-fp8-4steps`'s CI pattern exactly — same `DOCKER_USERNAME`/
`DOCKER_TOKEN` repo secrets, same disk-cleanup steps (this image's CUDA
12.9 base is large enough that the default GitHub-hosted runner needs the
extra space freed up before the build even starts pulling layers).

## RunPod Setup

Not yet deployed (out of scope for this scaffolding pass — see task
boundaries). Once built:

1. Create a new Serverless Endpoint (separate from the Wan2.2 workers' endpoints)
2. GPU: RTX 6000 Ada (47.4 GB) — see plan doc's VRAM budget section
3. Attach a network volume with the layout above
4. Set a generous timeout (video-gen jobs run minutes, not seconds)

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/runpod-volume/multitalk` | Root dir for base model + wav2vec2 + DiT checkpoints |
| `MULTITALK_WEIGHT_FORMAT` | unset (per-variant default: `fp8` for single, `fp8_lora` for multi) | Force a specific format for BOTH variants: `bf16` / `fp8` / `int8` / `int8_lora` (not `fp8_lora` — single has no such file) |
| `NUM_PERSISTENT_PARAM_IN_DIT` | `20000000000` | Passed through to InfiniteTalk's own low-VRAM paging flag (`--num_persistent_param_in_dit`). `0` streams every DiT layer CPU->GPU on every forward call (measured ~280s/step on a 47.5GB RTX 6000 Ada); the default is set above the ~14B-parameter DiT so the whole model stays GPU-resident instead. If this env var is set explicitly on the RunPod endpoint (e.g. still `0` from before this was raised), that overrides the code default — check the endpoint config, not just this table. |
| `USE_TEACACHE` | `1` | Enable InfiniteTalk's TeaCache speedup (`--use_teacache`) |
| `TEACACHE_THRESH` | `0.3` | TeaCache threshold, upstream-documented range 0.2-0.5 |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET_NAME` / `R2_PUBLIC_URL` | see `handler_v2.py` | Cloudflare R2 upload target for generated videos |

## Known gaps / what Phase 3 must verify

This is a **scaffolding pass** written without cloning or inspecting the
real InfiniteTalk repo (out of scope for this task — see
`MULTITALK-IMPLEMENTATION.md` Phase 3/6). Everything below is a best-effort
guess, clearly flagged at its point of use in the code:

- **`model_server.py`'s InfiniteTalk pipeline import** (`_import_pipeline_class()`)
  guesses at a module/class name (`wan.infinitetalk.InfiniteTalkPipeline` and
  two fallbacks) based on the Wan2.1/Wan2.2 code convention, not a confirmed
  fact. Fix `_PIPELINE_IMPORT_CANDIDATES` once the real repo is cloned.
- **The DiT submodule attribute name** (`self.pipe.model`, used by
  `load_dit()`'s swap) is assumed to match Wan2.1's `WanI2V.model`
  convention (no high/low-noise split, unlike Wan2.2's MoE).
- **`pipe.generate(...)`'s call signature** in `run_inference()` maps
  kwargs onto the CLI flags documented in the plan doc
  (`--sample_text_guide_scale`, `--cond_audio`, `--audio_type`, etc.) but
  the actual Python method signature is unverified.
- **The DiT quant-checkpoint loader** (`_load_dit_state_dict()`) assumes
  `safetensors.torch.load_file()` + `load_state_dict(strict=False)` is
  sufficient — i.e. that InfiniteTalk's own DiT module handles FP8/INT8
  tensors internally, unlike `wan22-14B-fp8-4steps`' custom `FP8Linear`
  dequant wrapper (deliberately **not** ported here per task scope — see
  plan doc's "do NOT port FP8Linear" framing).
- **Dockerfile**: no source patches applied (no `rope_apply` float64→float32
  downcast, no `wan/__init__.py` trim) — those were verified against
  Wan2.2's actual file layout in `wan22-14B-fp8-4steps` and have not been
  re-verified against InfiniteTalk's repo. Same classes of issue (VRAM,
  device-string bugs) may recur since InfiniteTalk is also Wan2.1-based.
- **`requirements.txt` audio deps** (`librosa`, `soundfile`, `opencv-python-headless`,
  `moviepy`, versions) are best-effort guesses for a wav2vec2-conditioned
  video-gen repo's typical needs, not confirmed against InfiniteTalk's own
  `requirements.txt`.
- **`duration_s` → `frame_num` formula and range** — see Parameters table
  above.
- **Checkpoint directory layout** (`format.json.example`) is inferred from
  the plan doc's description of InfiniteTalk's HF repo tree, not confirmed
  by an actual download.

None of this has been build-tested, pod-tested, or run against a GPU. Treat
`model_server.py` and the Dockerfile as a structured starting point for
Phase 3 (pod testing) and Phase 6 (Docker build & deploy), not working code.
# storystudio-multitalk
# storystudio-multitalk
