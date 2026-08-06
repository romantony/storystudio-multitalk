#!/usr/bin/env python
"""
Persistent Model Server for InfiniteTalk (audio-driven talking-avatar video).

Pattern: same persistent-process, Unix-socket-driven design as
~/wan22-14B-fp8-4steps/model_server.py — the model loads once on container
startup and stays warm in GPU memory between jobs, instead of a cold
load-per-request.

Architecture (see MULTITALK-IMPLEMENTATION.md, "Requirement: both
single-person and two-person modes"):

  InfiniteTalk ships SEPARATE checkpoint files for single-person vs.
  two-person (multi) generation — unlike MultiTalk's one unified file. This
  worker implements the plan doc's **option 2** design:

    - T5 text encoder, wav2vec2 audio encoder, and VAE stay resident in GPU
      (T5/VAE) or CPU (wav2vec2 — see below) memory for the life of the
      process (loaded once).
    - Only the DiT (`self.pipe.model`, a `WanModel`) gets swapped, via a
      meta-device-skeleton + `optimum.quanto.requantize()` reload (for the
      fp8/int8 quant checkpoints) when a job's person-count differs from
      whichever variant ("single" / "multi") is currently loaded. This is a
      few seconds (volume-throughput-bound), not a full cold start
      (~170-190s) — see ModelServer.load_dit().

  TODO — option 1, NOT implemented here: the plan doc's highest-priority
  open question is whether the `multi` checkpoint alone can serve
  single-person jobs at acceptable quality (populate only person1, omit
  person2). If confirmed, this entire swap mechanism collapses — load
  `multi` once at startup and never swap. That test requires a real GPU run
  and is out of scope for this pass. Option 2 (below) is the built-in
  fallback if option 1 doesn't hold.

CONFIRMED against the real MeiGen-AI/InfiniteTalk source (generate_infinitetalk.py
+ wan/multitalk.py, read in full):
  - `import wan; wan.InfiniteTalkPipeline` is the real pipeline class
    (`wan/__init__.py` does `from .multitalk import InfiniteTalkPipeline`).
  - The original crash (`ImportError: No module named 'xfuser'`) is
    explained, and `xfuser` IS a hard, unconditional dependency of
    `wan.multitalk` — NOT something gated behind `use_usp`. Cloned the real
    repo and grepped every file: `wan/utils/multitalk_utils.py` and
    `wan/modules/attention.py` both do a bare, MODULE-LEVEL
    `from xfuser.core.distributed import (...)` (not inside any `if
    use_usp:` guard). `multitalk.py` imports the first directly
    (`from .utils.multitalk_utils import MomentumBuffer, ...`, line ~32),
    and this file's own `_load_quant_dit_module()`/`_load_bf16_dit_module()`
    below import `wan.modules.multitalk_model`, which imports
    `.attention` (triggering the second) at module level too. So merely
    `import wan` — regardless of `use_usp` — requires `xfuser` to be
    installed; `requirements.txt`'s `xfuser>=0.4.1` (added alongside this
    fix, matching InfiniteTalk's own requirements.txt) is a real,
    non-optional dependency, not a defensive extra. (The `if use_usp:`
    branches elsewhere, e.g. multitalk.py ~line 250, only gate xfuser's
    *distributed init calls* — by the time you'd reach those, the module has
    already been imported regardless.)
  - The three DiT-loading branches inside `__init__` (quant / bf16-merge /
    pre-merged-dit_path) are ported below as standalone module-level
    helpers so `load_dit()` can redo the same dance for a checkpoint swap
    without re-running the whole pipeline constructor (which would also
    reload T5/VAE/CLIP unnecessarily).
  - wav2vec2 is NOT part of the pipeline class at all — it's loaded
    separately via `custom_init(device, wav2vec_dir)` and, per the real
    CLI (`custom_init('cpu', args.wav2vec_dir)`), lives on CPU permanently
    (all audio embedding extraction in `get_embedding()` also defaults to
    `device='cpu'`). Ported verbatim below.
  - `input_data['cond_video']` is the correct key even for a still image —
    InfiniteTalk treats an image as a 1-frame video via
    `extract_specific_frames()` / `is_video()`.
  - `generate_infinitetalk()`'s `extra_args` is read for exactly six
    attributes (grepped the full method body): `use_teacache`,
    `teacache_thresh`, `size` (used as `model_scale` for teacache_init),
    `use_apg`, `apg_momentum`, `apg_norm_threshold`. No other CLI-only
    attribute (e.g. `.mode`, `.scene_seg`) is read inside the method itself
    — `scene_seg`/`mode` are read by the CLI's own `generate()` driver
    function to decide clip-splitting/shot-detection *before* calling
    `generate_infinitetalk()`, not by the method. This worker always does
    single-clip generation (max_frames_num == frame_num, the CLI's default
    "clip" mode's effective behavior) — no shot-segmentation, no
    long-video streaming loop.
  - `size_buckget` (sic — genuine upstream typo, not ours) selects a
    bucket table (`ASPECT_RATIO_627` / `ASPECT_RATIO_960`) InfiniteTalk
    uses internally to pick the actual (H, W) from the input image's
    aspect ratio; the old `max_area`-based sizing this file used to pass
    was never a real parameter of the API.
  - `save_video_ffmpeg(video_tensor, save_path_without_ext, [audio_wav],
    fps=25, quality=5, high_quality_save=False)` appends ".mp4" itself and
    writes to `save_path_without_ext + ".mp4"` — ported the exact call
    below including the stem-stripping this requires.

STILL UNVERIFIED / NOT FULLY WIRED (left as TODOs at point of use):
  - The bf16 baseline DiT-loading branch (`_load_bf16_dit_module`) needs
    the base Wan2.1 7-shard `diffusion_pytorch_model-0000{1..7}-of-00007.safetensors`
    files under `checkpoint_dir`, which `scripts/download_base_encoders.py`
    does NOT currently fetch. This worker's default weight formats
    (fp8 / fp8_lora) never hit this branch, but it will fail if a job (or
    `MULTITALK_WEIGHT_FORMAT=bf16`) requests it until those shards are
    added to the download script.
  - The pre-merged single-file `dit_path` branch (`_load_premerged_dit_module`)
    is ported for completeness but not wired into `checkpoints`/weight_format
    routing — no "premerged" format exists in format.json yet, and we don't
    have any of those comfyui/infinitetalk_{single,multi}.safetensors files
    downloaded. Lower priority per task scope.
  - VRAM-management device placement when `NUM_PERSISTENT_PARAM_IN_DIT` is
    unset/None: per the real `__init__`, if `enable_vram_management()` is
    never called AND `init_on_cpu=True` (our fixed default, matches
    upstream — there's no CLI flag to change it), the DiT is simply never
    moved off CPU. This worker defaults `NUM_PERSISTENT_PARAM_IN_DIT=0`
    (not unset) specifically to avoid ever hitting that path in practice —
    flagged here since it's a faithful-but-surprising port of upstream
    behavior, not a bug we introduced.
"""
import gc
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

# Lightweight, non-CUDA-touching imports for the audio-preprocessing helpers
# ported from generate_infinitetalk.py — safe to import eagerly (mirrors
# that script's own top-of-file imports, which happen before any CUDA
# device selection since the CLI doesn't pin CUDA_VISIBLE_DEVICES itself).
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from einops import rearrange

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# IMPORTANT: must be set before any torch/wan import (below, and in every
# function that lazily imports them) — same discipline the file always had.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

INFINITETALK_PATH = "/workspace/infinitetalk"
if INFINITETALK_PATH not in sys.path:
    sys.path.insert(0, INFINITETALK_PATH)

MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/multitalk")
SOCKET_PATH = "/tmp/multitalk_model_server.sock"

TASK = "infinitetalk-14B"  # WAN_CONFIGS / SUPPORTED_SIZES key, confirmed via
                            # wan/configs/__init__.py

# --- Resolutions -------------------------------------------------------
# CONFIRMED: InfiniteTalk's `size` param is a bucket-name string
# ("infinitetalk-480" / "infinitetalk-720"), NOT a max_area int — the old
# RESOLUTIONS = {"480p": 832*480, ...} here was wrong. `shift` (flow-matching
# schedule) is a real, separate parameter the CLI hardcodes per size
# (_validate_args: 7 for -480, 11 for -720) — the old code never had it.
RESOLUTIONS = {
    "480p": "infinitetalk-480",
    "720p": "infinitetalk-720",
}
SHIFT_BY_SIZE = {
    "infinitetalk-480": 7.0,
    "infinitetalk-720": 11.0,
}

# --- Weight-format parameter deltas -------------------------------------
# Default (40-step, full-precision-equivalent) sampling vs. the lightx2v-
# merged 4-step FP8/INT8 "_lora" checkpoints. CONFIRMED sample_steps=40 /
# text_guide_scale=5.0 / audio_guide_scale=4.0 are generate_infinitetalk.py's
# own argparse defaults (--sample_steps None->40, --sample_text_guide_scale
# 5.0, --sample_audio_guide_scale 4.0). The 4-step lightx2v-LoRA values
# remain our own best-guess defaults (not in upstream argparse, no lightx2v
# recipe published for InfiniteTalk specifically) — override via per-job
# params if wrong.
WEIGHT_FORMAT_DEFAULTS = {
    "bf16":      {"sample_steps": 40, "text_guidance": 5.0, "audio_guidance": 4.0},
    "fp8":       {"sample_steps": 40, "text_guidance": 5.0, "audio_guidance": 4.0},
    "int8":      {"sample_steps": 40, "text_guidance": 5.0, "audio_guidance": 4.0},
    "fp8_lora":  {"sample_steps": 4,  "text_guidance": 1.0, "audio_guidance": 2.0},
    "int8_lora": {"sample_steps": 4,  "text_guidance": 1.0, "audio_guidance": 2.0},
}

# Default weight format served by this worker, PER variant — "single" has no
# officially-published fp8_lora file (verified 2026-08-06 against the live
# HF repo tree; only "multi" ships one), so the two variants default to
# different formats. See MULTITALK-IMPLEMENTATION.md "Update 2026-08-05" /
# "Update 2026-08-06". Override via MULTITALK_WEIGHT_FORMAT env var — if
# set, it applies to BOTH variants (must be a format that exists for both,
# i.e. not "fp8_lora"), overriding the per-variant defaults below.
_WEIGHT_FORMAT_ENV_OVERRIDE = os.getenv("MULTITALK_WEIGHT_FORMAT")
DEFAULT_WEIGHT_FORMAT_BY_VARIANT = (
    {"single": _WEIGHT_FORMAT_ENV_OVERRIDE, "multi": _WEIGHT_FORMAT_ENV_OVERRIDE}
    if _WEIGHT_FORMAT_ENV_OVERRIDE
    else {"single": "fp8", "multi": "fp8_lora"}
)

# Fallback checkpoint layout, used when MODEL_PATH/format.json is absent.
# Mirrors format.json.example — keep the two in sync.
#
# CONFIRMED: for quant formats, this path is the FULL PATH to the specific
# .safetensors file (passed as `quant_dir` to the pipeline / `_load_quant_dit_module`
# below) — matches multitalk.py's own `load_file(quant_dir)` /
# `quant_dir.replace('safetensors', 'json')` usage exactly. For "bf16", the
# path plays the role of `infinitetalk_dir` (the single audio-cross-attn
# weight file merged on top of the base Wan2.1 7-shard DiT — see
# _load_bf16_dit_module's TODO on the missing shards download).
_DEFAULT_FORMAT_CONFIG = {
    "format": "infinitetalk-v1",
    "base_dir": "Wan2.1-I2V-14B-480P",
    "wav2vec_dir": "chinese-wav2vec2-base",
    "checkpoints": {
        "single": {
            "bf16": "single/infinitetalk.safetensors",
            "fp8": "quant_models/infinitetalk_single_fp8.safetensors",
            "int8": "quant_models/infinitetalk_single_int8.safetensors",
            "int8_lora": "quant_models/infinitetalk_single_int8_lora.safetensors",
        },
        "multi": {
            "bf16": "multi/infinitetalk.safetensors",
            "fp8": "quant_models/infinitetalk_multi_fp8.safetensors",
            "fp8_lora": "quant_models/infinitetalk_multi_fp8_lora.safetensors",
            "int8": "quant_models/infinitetalk_multi_int8.safetensors",
            "int8_lora": "quant_models/infinitetalk_multi_int8_lora.safetensors",
        },
    },
    # Purely informational now — the real loading code derives the sidecar
    # json path itself via `quant_dir.replace('safetensors', 'json')`
    # (see _load_quant_dit_module), it doesn't consult this map. Kept as
    # documentation / for any download-side tooling that wants it.
    "checkpoint_sidecars": {
        "quant_models/infinitetalk_single_fp8.safetensors": "quant_models/infinitetalk_single_fp8.json",
        "quant_models/infinitetalk_single_int8.safetensors": "quant_models/infinitetalk_single_int8.json",
        "quant_models/infinitetalk_single_int8_lora.safetensors": "quant_models/infinitetalk_single_int8_lora.json",
        "quant_models/infinitetalk_multi_fp8.safetensors": "quant_models/infinitetalk_multi_fp8.json",
        "quant_models/infinitetalk_multi_fp8_lora.safetensors": "quant_models/infinitetalk_multi_fp8_lora.json",
        "quant_models/infinitetalk_multi_int8.safetensors": "quant_models/infinitetalk_multi_int8.json",
        "quant_models/infinitetalk_multi_int8_lora.safetensors": "quant_models/infinitetalk_multi_int8_lora.json",
    },
    "t5_fp8": "quant_models/t5_fp8.safetensors",
    "t5_fp8_map": "quant_models/t5_map_fp8.json",
    "quant_config": "quant_models/quant.json",
    "default_weight_format": dict(DEFAULT_WEIGHT_FORMAT_BY_VARIANT),
}


def _load_format_config() -> dict:
    fmt_file = Path(MODEL_PATH) / "format.json"
    if fmt_file.exists():
        cfg = json.loads(fmt_file.read_text())
        print(f"Loaded format config from {fmt_file}")
        return cfg
    print(f"No format.json at {fmt_file} — using built-in default layout "
          f"(see format.json.example)")
    return _DEFAULT_FORMAT_CONFIG


def _largest_4np1_at_most(n: int) -> int:
    """Largest value of the form 4k+1 that is <= n. CONFIRMED requirement
    (multitalk.py's generate_infinitetalk(), ~line 551-558): the frame mask
    is built by repeat_interleave-ing frame 0 four times and concatenating
    the rest (`torch.concat([repeat_interleave(msk[:,0:1], 4), msk[:,1:]])`),
    giving a temporal length of `frame_num + 3`, which then MUST divide
    evenly by 4 for `.view(1, T//4, 4, lat_h, lat_w)` to succeed — i.e.
    `frame_num % 4 == 1`. Also matches the CLI's own `--frame_num` help
    text ("The number should be 4n+1") and its default of 81 (=4*20+1). A
    RuntimeError "shape ... is invalid for input of size ..." at that
    .view() call is this constraint being violated.
    """
    if n < 1:
        raise ValueError(f"No valid 4n+1 frame count <= {n}")
    return 4 * ((n - 1) // 4) + 1


def _quant_kind_for_format(fmt: str) -> Optional[str]:
    """"fp8"/"fp8_lora" -> "fp8", "int8"/"int8_lora" -> "int8", "bf16" -> None.
    CONFIRMED: the `_lora`-suffixed quant files are loaded through the exact
    same `quant="fp8"`/`quant="int8"` constructor path as their plain
    counterparts — `InfiniteTalkPipeline.__init__` only accepts `quant` in
    {"int8", "fp8", None} (raises ValueError otherwise; multitalk.py line
    ~155), and `lora_dir` is explicitly skipped whenever `quant is not None`
    (line ~239), so the LoRA must already be baked into the requantized
    weights — there's no separate "fp8_lora" quant kind upstream."""
    if fmt.startswith("fp8"):
        return "fp8"
    if fmt.startswith("int8"):
        return "int8"
    if fmt == "bf16":
        return None
    raise ValueError(f"Unknown weight_format {fmt!r}")


# ════════════════════════════════════════════════════════════════════════════
# Audio preprocessing — ported near-verbatim from generate_infinitetalk.py
# (functions of the same name, module-level there too). Adapted only to
# drop the CLI `args` dependency in favor of explicit parameters.
# ════════════════════════════════════════════════════════════════════════════

def custom_init(device, wav2vec_dir):
    """CONFIRMED verbatim port of generate_infinitetalk.py's custom_init().
    Loads the wav2vec2 feature extractor + InfiniteTalk's custom
    Wav2Vec2Model (src.audio_analysis.wav2vec2, NOT the stock transformers
    class) — this is entirely separate from the InfiniteTalkPipeline class;
    the pipeline never touches wav2vec2 itself. Real CLI calls this with
    device='cpu' and never moves it — the audio encoder stays CPU-resident
    for the life of the process, matching get_embedding()'s own device='cpu'
    default below."""
    import torch  # noqa: F401 (imported for side effect parity / future use)
    from transformers import Wav2Vec2FeatureExtractor
    from src.audio_analysis.wav2vec2 import Wav2Vec2Model

    audio_encoder = Wav2Vec2Model.from_pretrained(
        wav2vec_dir, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_dir, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder


def loudness_norm(audio_array, sr=16000, lufs=-23):
    """Verbatim port."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    return pyln.normalize.loudness(audio_array, loudness, lufs)


def extract_audio_from_video(filename, sample_rate):
    """Verbatim port — used by audio_prepare_single() when a person's audio
    input is actually a video file (.mp4/.mov/.avi/.mkv)."""
    import subprocess

    raw_audio_path = filename.split('/')[-1].split('.')[0] + '.wav'
    ffmpeg_command = [
        "ffmpeg", "-y", "-i", str(filename),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)
    return human_speech_array


def audio_prepare_single(audio_path, sample_rate=16000):
    """Verbatim port."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        return extract_audio_from_video(audio_path, sample_rate)
    human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    return loudness_norm(human_speech_array, sr)


def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    """Verbatim port. `left_path`/`right_path` == 'None' (the literal string)
    signals that speaker is absent — not used by this worker today (our
    "multi" variant always has both person1 and person2 paths present by
    construction, see ModelServer._prepare_audio), but kept for fidelity."""
    if not (left_path == 'None' or right_path == 'None'):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path == 'None':
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path == 'None':
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type == 'para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type == 'add':
        new_human_speech1 = np.concatenate(
            [human_speech_array1, np.zeros(human_speech_array2.shape[0])])
        new_human_speech2 = np.concatenate(
            [np.zeros(human_speech_array1.shape[0]), human_speech_array2])
    else:
        raise ValueError(f"Unknown audio_type {audio_type!r} — only 'para' "
                          f"and 'add' are handled by audio_prepare_multi() "
                          f"upstream.")
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs


def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder,
                   sr=16000, device='cpu'):
    """Verbatim port. Returns a (seq_len, num_layers, hidden_dim) CPU
    tensor — this is what gets torch.save()'d to a .pt file and referenced
    by path in input_data['cond_audio'][...]."""
    import torch

    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # assume 25 fps

    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    with torch.no_grad():
        embeddings = audio_encoder(
            audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    return audio_emb.cpu().detach()


# ════════════════════════════════════════════════════════════════════════════
# DiT checkpoint loading — module-level helpers mirroring the three branches
# inside InfiniteTalkPipeline.__init__ (multitalk.py ~line 194-234), used by
# ModelServer.load_dit() to redo a checkpoint swap WITHOUT re-running the
# whole pipeline constructor (which would also reload T5/VAE/CLIP).
# ════════════════════════════════════════════════════════════════════════════

def _load_quant_dit_module(checkpoint_dir: str, quant_ckpt_path: str):
    """CONFIRMED verbatim port of the `quant is not None` branch
    (multitalk.py ~line 194-205): build a meta-device WanModel skeleton from
    `checkpoint_dir/config.json`, then `requantize()` the real quantized
    weights from `quant_ckpt_path` (a FULL PATH to one .safetensors file)
    onto it using the sidecar `.json` quantization map (same basename,
    `.safetensors` -> `.json`). Returns the quantized WanModel, on CPU
    (the `device='cpu'` in requantize() is copied verbatim from upstream —
    VRAM placement is handled afterwards by ModelServer.load_dit(), matching
    __init__'s own `enable_vram_management()` / `.to(device)` split)."""
    import torch
    from safetensors.torch import load_file
    from optimum.quanto import requantize
    from wan.modules.multitalk_model import WanModel

    with torch.device("meta"):
        wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
        model = WanModel(weight_init=False, **wan_config)
        torch.cuda.empty_cache()

    model_state_dict = load_file(quant_ckpt_path)
    map_json_path = quant_ckpt_path.replace("safetensors", "json")
    model.init_freqs()
    with open(map_json_path, "r") as f:
        quantization_map = json.load(f)
    requantize(model, model_state_dict, quantization_map, device="cpu")
    return model


def _load_bf16_dit_module(checkpoint_dir: str, infinitetalk_path: str, param_dtype):
    """CONFIRMED port of the `quant is None and dit_path is None` branch
    (multitalk.py ~line 207-224): merges the base Wan2.1 7-shard DiT
    safetensors under `checkpoint_dir` with the single InfiniteTalk
    audio-cross-attn weight file `infinitetalk_path` into one state_dict via
    plain load_state_dict (no meta-device trick here — matches upstream,
    which builds `self.model` as a real (non-meta) module for this branch;
    `init_contexts` is constructed upstream but never actually applied as a
    context manager for this branch, so we don't apply it here either — this
    looks like dead code in the real source, ported faithfully rather than
    "fixed").

    TODO — NOT YET DOWNLOADABLE: needs
    diffusion_pytorch_model-0000{1..7}-of-00007.safetensors under
    checkpoint_dir. scripts/download_base_encoders.py does not currently
    fetch these (only VAE/T5/CLIP encoder files + config.json) — this
    branch will raise FileNotFoundError until that's added. None of this
    worker's default weight formats (fp8 / fp8_lora) hit this branch."""
    from safetensors.torch import load_file
    from wan.modules.multitalk_model import WanModel

    wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
    model = WanModel(weight_init=False, **wan_config).to(dtype=param_dtype)

    weight_files = [
        f"{checkpoint_dir}/diffusion_pytorch_model-0000{i}-of-00007.safetensors"
        for i in range(1, 8)
    ] + [infinitetalk_path]
    merged_state_dict = {}
    for weight_file in weight_files:
        merged_state_dict.update(load_file(weight_file))
    model.load_state_dict(merged_state_dict)
    return model


def _load_premerged_dit_module(checkpoint_dir: str, dit_path: str):
    """Port of the `dit_path is not None` branch (multitalk.py ~line
    226-234) — a pre-merged single-file checkpoint (e.g. the repo's
    comfyui/infinitetalk_{single,multi}.safetensors). LOWER PRIORITY per
    task scope: implemented for completeness but NOT wired into
    `checkpoints`/weight_format routing (no "premerged" format exists in
    format.json yet, and none of these files are downloaded onto the
    volume). Call manually if this path becomes needed."""
    import torch
    from diffusers.models.modeling_utils import no_init_weights, ContextManagers
    import accelerate
    from wan.modules.multitalk_model import WanModel

    init_contexts = [no_init_weights(), accelerate.init_empty_weights()]
    with ContextManagers(init_contexts):
        wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
        model = WanModel(weight_init=False, **wan_config)
    checkpoint_weights = torch.load(dit_path, map_location="cpu")
    model.load_state_dict(checkpoint_weights["state_dict"])
    return model


class ModelServer:
    """Holds one resident InfiniteTalk pipeline (T5 + VAE + CLIP + DiT) plus
    the separately-loaded wav2vec2 audio encoder. T5/VAE/CLIP/wav2vec2 never
    change after load_model(). The DiT (self.pipe.model) is swapped in-place
    by load_dit() when a job needs a different person-count variant than
    whatever's currently loaded.
    """

    def __init__(self):
        self.pipe = None                    # wan.InfiniteTalkPipeline instance
        self.wav2vec_feature_extractor = None
        self.audio_encoder = None
        self.current_dit_variant: Optional[str] = None  # "single" | "multi"
        self.weight_format: Optional[str] = None  # format of the currently-loaded DiT
        self.format_config = None
        self.device = None

        # CONFIRMED: NOT a constructor kwarg — called separately via
        # pipe.enable_vram_management(num_persistent_param_in_dit=N) after
        # construction (see load_model()/load_dit()). Real CLI only calls
        # it `if args.num_persistent_param_in_dit is not None`; we default
        # to "0" (aggressive offload, everything else onloaded per-layer
        # during forward) rather than leaving it unset, since with
        # init_on_cpu=True (our fixed default, matches upstream — no CLI
        # flag changes it) an unset value would leave the DiT stranded on
        # CPU with no VRAM management to move it. See module docstring.
        _npp_env = os.getenv("NUM_PERSISTENT_PARAM_IN_DIT", "0")
        self.num_persistent_param_in_dit = (
            None if _npp_env.strip().lower() in ("", "none") else int(_npp_env)
        )

        self.use_teacache = os.getenv("USE_TEACACHE", "1") not in ("0", "false", "False")
        self.teacache_thresh = float(os.getenv("TEACACHE_THRESH", "0.2"))  # CLI default 0.2
        self.use_apg = os.getenv("USE_APG", "0") not in ("0", "false", "False")
        self.apg_momentum = float(os.getenv("APG_MOMENTUM", "-0.75"))  # CLI default
        self.apg_norm_threshold = float(os.getenv("APG_NORM_THRESHOLD", "55"))  # CLI default
        self._timings = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _default_format_for(self, variant: str) -> str:
        """Per-variant default weight format — "single" has no officially
        published fp8_lora file, so it defaults to plain "fp8" while "multi"
        defaults to "fp8_lora". See DEFAULT_WEIGHT_FORMAT_BY_VARIANT."""
        default_map = self.format_config.get(
            "default_weight_format", DEFAULT_WEIGHT_FORMAT_BY_VARIANT)
        if isinstance(default_map, str):  # tolerate an old-style single-string format.json
            return default_map
        return default_map[variant]

    def _checkpoint_path(self, variant: str, weight_format: Optional[str] = None) -> str:
        """Resolve the absolute on-disk path for a (variant, weight_format)
        pair using the loaded format config (format.json or built-in
        default — see _load_format_config / format.json.example)."""
        fmt = weight_format or self._default_format_for(variant)
        try:
            rel = self.format_config["checkpoints"][variant][fmt]
        except KeyError:
            raise ValueError(
                f"No checkpoint configured for variant={variant!r} "
                f"weight_format={fmt!r} in format.json / default layout. "
                f"Known variants: {list(self.format_config['checkpoints'])}, "
                f"known formats for 'single': "
                f"{list(self.format_config['checkpoints'].get('single', {}))}"
            )
        return str(Path(MODEL_PATH) / rel)

    def _base_dir(self) -> str:
        return str(Path(MODEL_PATH) / self.format_config["base_dir"])

    def load_model(self):
        """Load T5 + CLIP + VAE + the default DiT variant ("single") via the
        real InfiniteTalkPipeline constructor, plus wav2vec2 (loaded
        separately — see custom_init()). Everything except the DiT stays
        resident for the life of the process; see load_dit() for how the
        DiT itself gets swapped later.
        """
        import torch

        print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
        try:
            import flash_attn
            print(f"FlashAttention 2 available: v{flash_attn.__version__} (real flash kernel)")
        except Exception:
            print("FlashAttention NOT installed — attention falls back to PyTorch SDPA")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f} GB")
        self.device = torch.device("cuda", 0)

        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}.")

        self.format_config = _load_format_config()
        self.weight_format = self._default_format_for("single")
        defaults = WEIGHT_FORMAT_DEFAULTS.get(
            self.weight_format, WEIGHT_FORMAT_DEFAULTS["fp8"])
        print(f"Weight format: {self.weight_format} "
              f"(default sample_steps={defaults['sample_steps']}, "
              f"text_guidance={defaults['text_guidance']}, "
              f"audio_guidance={defaults['audio_guidance']})")

        base_dir = self._base_dir()
        wav2vec_dir = str(Path(MODEL_PATH) / self.format_config["wav2vec_dir"])

        print(f"Loading InfiniteTalk pipeline (base={base_dir}, "
              f"wav2vec={wav2vec_dir}) — this loads T5/VAE/CLIP (resident) "
              f"plus the initial 'single' DiT ...")
        start = time.time()

        import wan
        from wan.configs import WAN_CONFIGS
        cfg = WAN_CONFIGS[TASK]

        quant_kind = _quant_kind_for_format(self.weight_format)
        if quant_kind is not None:
            quant_dir_kwarg = self._checkpoint_path("single", self.weight_format)
            infinitetalk_dir_kwarg = None
        else:  # "bf16" — see _load_bf16_dit_module's TODO on missing shards
            quant_dir_kwarg = None
            infinitetalk_dir_kwarg = self._checkpoint_path("single", "bf16")

        # CONFIRMED constructor kwargs (multitalk.py __init__ signature +
        # generate_infinitetalk.py's own `wan.InfiniteTalkPipeline(...)` call
        # site): use_usp=False keeps this single-GPU worker off the xfuser
        # import path entirely (see module docstring).
        self.pipe = wan.InfiniteTalkPipeline(
            config=cfg,
            checkpoint_dir=base_dir,
            quant_dir=quant_dir_kwarg,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=True,
            init_on_cpu=True,
            lora_dir=None,
            lora_scales=None,
            quant=quant_kind,
            dit_path=None,
            infinitetalk_dir=infinitetalk_dir_kwarg,
        )
        self.current_dit_variant = "single"

        if self.num_persistent_param_in_dit is not None:
            self.pipe.vram_management = True
            self.pipe.enable_vram_management(
                num_persistent_param_in_dit=self.num_persistent_param_in_dit)

        print("Loading wav2vec2 (CPU-resident, per upstream custom_init('cpu', ...))...")
        self.wav2vec_feature_extractor, self.audio_encoder = custom_init("cpu", wav2vec_dir)

        self._instrument_timing()

        elapsed = time.time() - start
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"✓ Model ready in {elapsed:.1f}s — {allocated:.1f} GB alloc / "
              f"{reserved:.1f} GB reserved (variant={self.current_dit_variant})")

    def _instrument_timing(self) -> None:
        """Best-effort per-phase timing wrapper, matching
        ~/wan22-14B-fp8-4steps/model_server.py's _instrument_timing(). Skips
        silently (with a warning) if self.pipe doesn't expose the expected
        `.vae` attribute — cosmetic only, doesn't block generation. Note:
        only vae_encode/vae_decode are wrapped (t5/wav2vec entries in
        self._timings stay 0.0 always) — same as before this rewrite."""
        self._timings = {"t5": 0.0, "wav2vec": 0.0, "vae_encode": 0.0, "vae_decode": 0.0}
        try:
            vae = self.pipe.vae
            _oenc, _odec = vae.encode, vae.decode

            def _timed_encode(*a, **k):
                s = time.time(); r = _oenc(*a, **k)
                self._timings["vae_encode"] += time.time() - s
                return r

            def _timed_decode(*a, **k):
                s = time.time(); r = _odec(*a, **k)
                self._timings["vae_decode"] += time.time() - s
                return r

            vae.encode, vae.decode = _timed_encode, _timed_decode
        except AttributeError as e:
            print(f"    [timing] Skipping VAE timing wrap — unexpected pipe "
                  f"attribute layout ({e}). Cosmetic only.")

    # ------------------------------------------------------------------
    # DiT swap (option 2)
    # ------------------------------------------------------------------

    def load_dit(self, variant: str, weight_format: Optional[str] = None) -> None:
        """Swap the resident DiT (`self.pipe.model`) to `variant` ("single"
        or "multi") if it isn't already loaded. T5/VAE/CLIP/wav2vec2 are
        untouched. Redoes the exact meta-skeleton + requantize() (or bf16
        merge) dance InfiniteTalkPipeline.__init__ does for its OWN initial
        DiT load — see _load_quant_dit_module / _load_bf16_dit_module.
        """
        import torch

        fmt = weight_format or self._default_format_for(variant)
        if variant == self.current_dit_variant and fmt == self.weight_format:
            return

        ckpt_path = self._checkpoint_path(variant, fmt)
        print(f"[dit-swap] {self.current_dit_variant!r} -> {variant!r} "
              f"(format={fmt}): loading {ckpt_path}")
        t0 = time.time()

        quant_kind = _quant_kind_for_format(fmt)
        if quant_kind is not None:
            new_model = _load_quant_dit_module(self._base_dir(), ckpt_path)
        elif fmt == "bf16":
            new_model = _load_bf16_dit_module(
                self._base_dir(), ckpt_path, self.pipe.param_dtype)
        else:
            raise ValueError(f"Unsupported weight_format for DiT swap: {fmt!r}")

        # CONFIRMED: mirrors the common tail of __init__ that runs after
        # both loading branches (multitalk.py ~line 236-238) — unconditional
        # regardless of quant vs bf16.
        new_model.eval().requires_grad_(False)
        from wan.multitalk import to_param_dtype_fp32only
        to_param_dtype_fp32only(new_model, self.pipe.param_dtype)

        old_model = self.pipe.model
        self.pipe.model = new_model
        del old_model
        gc.collect()
        torch.cuda.empty_cache()

        if self.num_persistent_param_in_dit is not None:
            self.pipe.vram_management = True
            self.pipe.enable_vram_management(
                num_persistent_param_in_dit=self.num_persistent_param_in_dit)
        else:
            # Matches __init__'s `if not init_on_cpu: self.model.to(device)`
            # — our init_on_cpu is always True, so upstream would actually
            # leave this on CPU here too. We move it explicitly instead
            # since NUM_PERSISTENT_PARAM_IN_DIT unset is not our supported
            # configuration (see module docstring) and stranding the DiT on
            # CPU would silently break inference.
            new_model.to(self.device)

        self.current_dit_variant = variant
        self.weight_format = fmt
        print(f"[dit-swap] done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _prepare_audio(self, person_audio_paths: dict, audio_type: str,
                        variant: str, tmp_dir: Path):
        """Port of generate_infinitetalk.py's generate()'s inline audio-prep
        block (lines ~600-624 of the CLI): raw wav -> audio_prepare_*() ->
        get_embedding() -> torch.save() to a .pt file, PLUS a mixed-down
        "video_audio" wav used later for final muxing. Returns
        (cond_audio_dict, video_audio_path, min_emb_len) ready to drop into
        input_data — min_emb_len is the shortest embedding's shape[0]
        (frame-equivalent length) across all speakers, used by run_inference
        to clamp frame_num (see its CONFIRMED note on multitalk.py's
        `full_audio_emb.shape[0] <= frame_num: continue` filter).
        """
        import torch

        if variant == "multi":
            new_h1, new_h2, summed = audio_prepare_multi(
                person_audio_paths["person1"], person_audio_paths["person2"],
                audio_type)
            emb1 = get_embedding(new_h1, self.wav2vec_feature_extractor, self.audio_encoder)
            emb2 = get_embedding(new_h2, self.wav2vec_feature_extractor, self.audio_encoder)
            emb1_path = str(tmp_dir / "1.pt")
            emb2_path = str(tmp_dir / "2.pt")
            torch.save(emb1, emb1_path)
            torch.save(emb2, emb2_path)
            video_audio_path = str(tmp_dir / "sum.wav")
            sf.write(video_audio_path, summed, 16000)
            min_emb_len = min(emb1.shape[0], emb2.shape[0])
            return {"person1": emb1_path, "person2": emb2_path}, video_audio_path, min_emb_len

        human_speech = audio_prepare_single(person_audio_paths["person1"])
        emb = get_embedding(human_speech, self.wav2vec_feature_extractor, self.audio_encoder)
        emb_path = str(tmp_dir / "1.pt")
        torch.save(emb, emb_path)
        video_audio_path = str(tmp_dir / "sum.wav")
        sf.write(video_audio_path, human_speech, 16000)
        return {"person1": emb_path}, video_audio_path, emb.shape[0]

    def run_inference(self, params: dict) -> dict:
        """Run one generation job. `params` mirrors the request dict built
        by handler_v2.py's send_to_model_server() call — that internal
        socket-protocol shape is unchanged by this rewrite (image_path,
        person_audio_paths, etc.); only what we DO with it below changed.

        CONFIRMED port: builds the real `input_data`/`extra_args` shapes
        (see module docstring) and calls
        `self.pipe.generate_infinitetalk(...)` — the real DiT forward /
        L-RoPE / audio cross-attn sampling loop lives entirely inside that
        method (multitalk.py), we don't reimplement any of it.
        """
        import shutil

        image_path = params["image_path"]
        prompt = params.get("prompt", "")
        person_audio_paths = params["person_audio_paths"]  # {"person1": path, "person2": path?}
        audio_type = params.get("audio_type", "para")
        resolution = params.get("resolution", "480p")
        frame_num = params.get("frame_num", 81)
        output_path = params["output_path"]

        variant = "multi" if "person2" in person_audio_paths else "single"

        if resolution not in RESOLUTIONS:
            raise ValueError(f"Unknown resolution '{resolution}'. Choose 480p or 720p.")
        size_bucket = RESOLUTIONS[resolution]
        shift = params.get("shift", SHIFT_BY_SIZE[size_bucket])

        # Swap DiT if this job's person-count doesn't match what's resident.
        # Must happen BEFORE reading self.weight_format below — load_dit()
        # updates it to whatever format the target variant actually loaded
        # in (e.g. "single" defaults to "fp8", "multi" to "fp8_lora"; see
        # DEFAULT_WEIGHT_FORMAT_BY_VARIANT). Reading it before the swap would
        # use the PREVIOUSLY loaded variant's format/step-count defaults.
        self.load_dit(variant)

        defaults = WEIGHT_FORMAT_DEFAULTS.get(
            self.weight_format, WEIGHT_FORMAT_DEFAULTS["fp8"])
        sample_steps = params.get("sample_steps", defaults["sample_steps"])
        text_guidance = params.get("text_guidance_scale", defaults["text_guidance"])
        audio_guidance = params.get("audio_guidance_scale", defaults["audio_guidance"])
        motion_frame = params.get("motion_frame", 9)  # CLI's real default (9), not the
                                                        # method's own signature default (25)

        print(f"Generating {resolution} (variant={variant}, size={size_bucket}), "
              f"{frame_num} frames, {sample_steps} steps, shift={shift}, "
              f"text_guidance={text_guidance}, audio_guidance={audio_guidance}, "
              f"audio_type={audio_type}, teacache={self.use_teacache}")

        self._timings = {"t5": 0.0, "wav2vec": 0.0, "vae_encode": 0.0, "vae_decode": 0.0}

        tmp_dir = Path(output_path).parent / f"_infinitetalk_audio_{params.get('job_id', 'job')}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            cond_audio, video_audio_path, min_emb_len = self._prepare_audio(
                person_audio_paths, audio_type, variant, tmp_dir)

            # CONFIRMED (multitalk.py generate_infinitetalk() line ~478):
            # `if full_audio_emb.shape[0] <= frame_num: continue` — an audio
            # embedding is SILENTLY DROPPED if its length isn't STRICTLY
            # GREATER than frame_num, and the method then asserts every
            # speaker's embedding survived that filter (line ~482, "Aduio
            # file not exists or length not satisfies frame nums." — real
            # upstream typo). Our old frame_num math (duration_s*25 + 1)
            # guaranteed this assert whenever the source audio was close to
            # duration_s seconds long, since it asked for MORE frames than
            # a duration_s-long clip naturally provides. Clamp here using
            # the ACTUAL embedding length instead of trusting the
            # client-requested frame_num blindly.
            if frame_num >= min_emb_len:
                clamped = max(1, min_emb_len - 1)
                print(f"[frame_num] requested {frame_num} >= audio embedding "
                      f"length {min_emb_len} — clamping to {clamped} "
                      f"(upstream requires audio strictly longer than "
                      f"frame_num, see multitalk.py's HUMAN_NUMBER assert)")
                frame_num = clamped

            # CONFIRMED (multitalk.py ~line 551-558): frame_num MUST be of
            # the form 4n+1 or the frame-mask .view() call crashes with
            # "shape [...] is invalid for input of size [...]" — see
            # _largest_4np1_at_most's docstring. Round DOWN only (never up,
            # which could put us back over the audio-length clamp above).
            normalized = _largest_4np1_at_most(frame_num)
            if normalized != frame_num:
                print(f"[frame_num] {frame_num} is not of the form 4n+1 "
                      f"(required by generate_infinitetalk's frame-mask "
                      f"reshape) — rounding down to {normalized}")
                frame_num = normalized

            # CONFIRMED shape (multitalk.py generate_infinitetalk() reads
            # exactly these keys): 'cond_video' (NOT 'cond_image', even for
            # a still image — see module docstring), 'cond_audio' (dict of
            # embedding .pt PATHS, not tensors/raw wav), 'audio_type',
            # 'video_audio' (mixed-down wav path used only for final mux).
            input_data = {
                "prompt": prompt,
                "cond_video": image_path,
                "cond_audio": cond_audio,
                "audio_type": audio_type,
                "video_audio": video_audio_path,
            }

            # CONFIRMED: exactly these 6 attributes are read off extra_args
            # inside generate_infinitetalk() (grepped the full method body —
            # no `.mode`/`.scene_seg`/other CLI-only attrs are touched here).
            extra_args = SimpleNamespace(
                use_teacache=self.use_teacache,
                teacache_thresh=self.teacache_thresh,
                size=size_bucket,
                use_apg=self.use_apg,
                apg_momentum=self.apg_momentum,
                apg_norm_threshold=self.apg_norm_threshold,
            )

            # max_frames_num=frame_num forces the internal while-loop
            # (`if max_frames_num <= frame_num: break`) to exit after one
            # iteration — i.e. the CLI's default "clip" mode behavior
            # (single clip, no long-video streaming/motion-frame carry-over
            # across chunks). We don't implement streaming mode.
            video_tensor = self.pipe.generate_infinitetalk(
                input_data,
                size_buckget=size_bucket,  # sic — real upstream kwarg name (typo)
                motion_frame=motion_frame,
                frame_num=frame_num,
                shift=shift,
                sampling_steps=sample_steps,
                text_guide_scale=text_guidance,
                audio_guide_scale=audio_guidance,
                n_prompt=params.get("negative_prompt", ""),
                seed=params.get("seed", -1),
                offload_model=params.get("offload_model", True),
                max_frames_num=frame_num,
                color_correction_strength=params.get("color_correction_strength", 0.0),
                extra_args=extra_args,
            )

            self._write_video(video_tensor, video_audio_path, output_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return {"variant": variant, "sample_steps": sample_steps, "frame_num": frame_num}

    def _write_video(self, video_tensor, video_audio_path: str, output_path: str) -> None:
        """CONFIRMED port of generate_infinitetalk.py's final muxing call:
        `save_video_ffmpeg(sum_video, save_file_without_ext, [video_audio_path],
        high_quality_save=False)`. save_video_ffmpeg() appends ".mp4" itself
        and writes to `save_path + ".mp4"`, so we pass the extension-stripped
        stem and rename the result back to our own `output_path` if it
        differs."""
        from wan.utils.multitalk_utils import save_video_ffmpeg

        output_path_str = str(output_path)
        save_stem = (output_path_str[:-4] if output_path_str.lower().endswith(".mp4")
                     else output_path_str)

        save_video_ffmpeg(
            video_tensor, save_stem, [video_audio_path],
            fps=25, quality=5, high_quality_save=False)

        produced = save_stem + ".mp4"
        if produced != output_path_str:
            os.replace(produced, output_path_str)

        if not Path(output_path_str).exists():
            raise RuntimeError(f"Video not created at {output_path_str}")

    def generate(self, params: dict) -> dict:
        """Top-level entry point called from the socket loop. Wraps
        run_inference() with timing + error handling + post-failure VRAM
        cleanup, matching ~/wan22-14B-fp8-4steps/model_server.py's
        generate_video()."""
        import torch

        start = time.time()
        try:
            output_path = params["output_path"]
            extra = self.run_inference(params)

            size_mb = Path(output_path).stat().st_size / 1024 / 1024
            gen_time = time.time() - start

            t = self._timings
            print(f"✓ Done in {gen_time:.1f}s — {size_mb:.1f} MB")
            print(f"  timing: t5={t.get('t5', 0):.1f}s | "
                  f"wav2vec={t.get('wav2vec', 0):.1f}s | "
                  f"vae_encode={t.get('vae_encode', 0):.1f}s | "
                  f"vae_decode={t.get('vae_decode', 0):.1f}s")

            return {
                "success": True,
                "output_path": output_path,
                "generation_time": gen_time,
                "file_size_mb": round(size_mb, 2),
                **extra,
            }
        except Exception as e:
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"    Post-failure cleanup: {free_gb:.1f} GB free after empty_cache()")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Socket server loop
    # ------------------------------------------------------------------

    def run(self):
        import signal

        def shutdown(signum, frame):
            print(f"Signal {signum} — shutting down")
            sys.exit(0)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        self.load_model()

        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(1)
        os.chmod(SOCKET_PATH, 0o777)

        print(f"✓ Model server ready on {SOCKET_PATH}")

        while True:
            conn = None
            try:
                conn, _ = server.accept()
                conn.settimeout(5.0)

                data = b""
                try:
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                        if b"\n\n" in data:
                            break
                except socket.timeout:
                    pass

                request_str = data.decode().strip()
                if not request_str:
                    conn.close()
                    continue

                request = json.loads(request_str)
                print(f"\nJob {request.get('job_id', 'unknown')}")

                result = self.generate(request)
                conn.sendall((json.dumps(result) + "\n").encode())

            except Exception as e:
                print(f"Server error: {e}")
                traceback.print_exc()
                if conn:
                    try:
                        conn.sendall(
                            (json.dumps({"success": False, "error": str(e)}) + "\n").encode()
                        )
                    except Exception:
                        pass
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass


if __name__ == "__main__":
    ModelServer().run()
