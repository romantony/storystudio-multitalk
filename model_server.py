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
      memory for the life of the process (loaded once).
    - Only the ~19.5 GB DiT safetensors checkpoint gets swapped, via a
      `load_state_dict`-style reload, when a job's person-count differs from
      whichever variant ("single" / "multi") is currently loaded. This is a
      few seconds (volume-throughput-bound), not a full cold start
      (~170-190s) — see ModelServer.load_dit() / _swap_dit_weights().

  TODO — option 1, NOT implemented here: the plan doc's highest-priority
  open question is whether the `multi` checkpoint alone can serve
  single-person jobs at acceptable quality (populate only person1, omit
  person2). If confirmed in Phase 3, this entire swap mechanism collapses —
  load `multi` once at startup and never swap. That test requires a real
  GPU + the real repo and is explicitly out of scope for this scaffolding
  pass. Option 2 (below) is the built-in fallback if option 1 doesn't hold.

IMPORTANT — best-effort port, not verified:
  The InfiniteTalk repo (github.com/MeiGen-AI/InfiniteTalk) was NOT cloned
  or inspected while writing this file (out of scope for this scaffolding
  pass — see task boundaries). Its exact Python package layout (module
  paths, class names, pipeline method signatures) is UNKNOWN. Everything
  below that touches InfiniteTalk's own code is a best-effort guess modeled
  on:
    (a) the CLI flags/`input_json` schema documented in
        MULTITALK-IMPLEMENTATION.md's "Acceleration" / "API Design"
        sections (--ckpt_dir, --wav2vec_dir, --quant_dir, --lora_dir,
        --sample_steps, --sample_text_guide_scale, --sample_audio_guide_scale,
        --mode, --num_persistent_param_in_dit, --use_teacache), and
    (b) the Wan2.1/Wan2.2 code convention that ~/wan22-14B-fp8-4steps
        already confirms works (a thin generate.py CLI wrapping an
        importable `wan.WanI2V`-style pipeline class; T5/VAE modules at
        `wan.modules.t5` / `wan.modules.vae`) — InfiniteTalk forked from
        that same Wan2.1 lineage, so this is a reasonable starting guess,
        not a verified fact.
  Every guess is called out at its point of use below with a comment
  explaining exactly what to check/fix once the real repo is available
  (Phase 3). The seams (load_dit(), run_inference(), _import_dit_class())
  are structured so fixing a wrong guess is a local, one-function change.
"""
import gc
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

INFINITETALK_PATH = "/workspace/infinitetalk"
if INFINITETALK_PATH not in sys.path:
    sys.path.insert(0, INFINITETALK_PATH)

MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/multitalk")
SOCKET_PATH = "/tmp/multitalk_model_server.sock"

# --- Resolutions -------------------------------------------------------
# 720p single-GPU support is an open question (MULTITALK-IMPLEMENTATION.md
# Open Question 3 — upstream README described it as multi-GPU-only /
# "update forthcoming" at doc-write time). Exposed here for completeness;
# handler_v2.py may choose to reject it until confirmed in Phase 3.
RESOLUTIONS = {
    "480p": 832 * 480,
    "720p": 1280 * 720,
}

# --- Weight-format parameter deltas -------------------------------------
# Default (40-step, full-precision-equivalent) sampling vs. the lightx2v-
# merged 4-step FP8/INT8 "_lora" checkpoints. Mirrors the pattern learned
# tuning wan22-14B-fp8-4steps: CFG-distilled / step-distilled weights need
# different guidance scales, not just fewer steps.
#
# UNCONFIRMED (MULTITALK-IMPLEMENTATION.md "What's still unverified"):
# whether the `_lora`-suffixed quant files are really 4-step-ready with just
# `--quant_dir` (no separate `--lora_dir`), given they're byte-identical in
# size to their non-`_lora` counterparts (strong circumstantial evidence the
# LoRA is baked in, but not confirmed at inference time). These are our
# best-guess defaults per weight format; override via env or per-job params.
WEIGHT_FORMAT_DEFAULTS = {
    "bf16":      {"sample_steps": 40, "text_guidance": 5.0, "audio_guidance": 4.0},
    "fp8":       {"sample_steps": 40, "text_guidance": 5.0, "audio_guidance": 4.0},
    "int8":      {"sample_steps": 40, "text_guidance": 5.0, "audio_guidance": 4.0},
    "fp8_lora":  {"sample_steps": 4,  "text_guidance": 1.0, "audio_guidance": 2.0},
    "int8_lora": {"sample_steps": 4,  "text_guidance": 1.0, "audio_guidance": 2.0},
}

# Default weight format served by this worker. fp8_lora = InfiniteTalk's
# officially-published pre-quantized, (probably) LoRA-merged single-file FP8
# DiT — see MULTITALK-IMPLEMENTATION.md "Update 2026-08-05". Override via
# MULTITALK_WEIGHT_FORMAT env var (one of WEIGHT_FORMAT_DEFAULTS' keys).
DEFAULT_WEIGHT_FORMAT = os.getenv("MULTITALK_WEIGHT_FORMAT", "fp8_lora")

# Fallback checkpoint layout, used when MODEL_PATH/format.json is absent.
# Mirrors format.json.example — keep the two in sync.
_DEFAULT_FORMAT_CONFIG = {
    "format": "infinitetalk-v1",
    "base_dir": "Wan2.1-I2V-14B-480P",
    "wav2vec_dir": "chinese-wav2vec2-base",
    "checkpoints": {
        "single": {
            "bf16": "single/infinitetalk.safetensors",
            "fp8": "quant_models/infinitetalk_single_fp8.safetensors",
            "fp8_lora": "quant_models/infinitetalk_single_fp8_lora.safetensors",
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
    "t5_fp8": "quant_models/t5_fp8.safetensors",
    "default_weight_format": DEFAULT_WEIGHT_FORMAT,
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


# ════════════════════════════════════════════════════════════════════════════
# Best-effort imports for InfiniteTalk's pipeline internals — GUESSES.
# See module docstring. Fix these once the real repo is cloned (Phase 3).
# ════════════════════════════════════════════════════════════════════════════

# Candidate (module, class) pairs for InfiniteTalk's audio-conditioned DiT /
# top-level pipeline class. Following the Wan-family rename pattern the plan
# doc gives (generate_multitalk.py -> generate_infinitetalk.py), the most
# likely name is "InfiniteTalkPipeline" in a module named after the project,
# but we try a couple of plausible alternates too.
_PIPELINE_IMPORT_CANDIDATES = [
    ("wan.infinitetalk", "InfiniteTalkPipeline"),
    ("wan.multitalk", "InfiniteTalkPipeline"),
    ("infinitetalk", "InfiniteTalkPipeline"),
]


def _import_pipeline_class():
    """Best-effort import of InfiniteTalk's top-level pipeline class (the
    analogue of wan.image2video.WanI2V in ~/wan22-14B-fp8-4steps). Tries
    each candidate module path in turn; first successful import wins.

    NOT VERIFIED — update _PIPELINE_IMPORT_CANDIDATES with the real
    module/class name once the InfiniteTalk repo is inspected. Check
    generate_infinitetalk.py's own top-of-file imports for the ground truth.
    """
    errors = []
    for module_name, class_name in _PIPELINE_IMPORT_CANDIDATES:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"Resolved InfiniteTalk pipeline class: {module_name}.{class_name}")
            return cls
        except Exception as e:  # noqa: BLE001 - collecting for diagnostics
            errors.append(f"{module_name}.{class_name}: {e}")
    raise ImportError(
        "Could not import InfiniteTalk's pipeline class from any candidate "
        "module path. This is EXPECTED until the real InfiniteTalk repo is "
        "cloned and inspected (see model_server.py module docstring) — "
        "update _PIPELINE_IMPORT_CANDIDATES with the correct module/class "
        "name from generate_infinitetalk.py's own imports. Errors seen: "
        + "; ".join(errors)
    )


# ════════════════════════════════════════════════════════════════════════════
# DiT checkpoint swap (option 2) — load_state_dict-style reload
# ════════════════════════════════════════════════════════════════════════════

def _load_dit_state_dict(checkpoint_path: str) -> dict:
    """Load a single-file InfiniteTalk DiT checkpoint (bf16, fp8, or int8)
    into a plain state_dict.

    We deliberately do NOT reimplement anything like wan22-14B-fp8-4steps'
    FP8Linear wrapper here — per the task scope, InfiniteTalk's FP8/INT8
    quant formats come as pre-quantized single-file checkpoints from
    HuggingFace and (per the plan doc) are expected to be loaded via
    InfiniteTalk's own --quant_dir code path, not a custom dequant wrapper.
    safetensors.torch.load_file returns whatever dtype each tensor was saved
    as (float8_e4m3fn tensors included, torch >= 2.1 supports this natively)
    — the DiT module's own load_state_dict is assumed to handle placing
    those tensors correctly. If InfiniteTalk's real quant format instead
    needs a custom loader (e.g. per-channel scale tensors under a different
    key convention, as lightx2v's Wan2.2 FP8 format does), port the relevant
    loader from ~/wan22-14B-fp8-4steps/model_server.py
    (_load_lightx2v_fp8_model) once the real safetensors header is
    inspected.
    """
    from safetensors.torch import load_file
    return load_file(checkpoint_path)


class ModelServer:
    """Holds one resident InfiniteTalk pipeline (T5 + wav2vec2 + VAE + DiT).
    T5/wav2vec2/VAE never change after load_model(). The DiT is swapped
    in-place by load_dit() when a job needs a different person-count variant
    than whatever's currently loaded.
    """

    def __init__(self):
        self.pipe = None                    # InfiniteTalk pipeline instance
        self.current_dit_variant: Optional[str] = None  # "single" | "multi"
        self.weight_format = DEFAULT_WEIGHT_FORMAT
        self.format_config = None
        self.device = None
        self.num_persistent_param_in_dit = int(
            os.getenv("NUM_PERSISTENT_PARAM_IN_DIT", "0"))
        self.use_teacache = os.getenv("USE_TEACACHE", "1") not in ("0", "false", "False")
        self.teacache_thresh = float(os.getenv("TEACACHE_THRESH", "0.3"))
        self._timings = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _checkpoint_path(self, variant: str, weight_format: Optional[str] = None) -> str:
        """Resolve the absolute on-disk path for a (variant, weight_format)
        pair using the loaded format config (format.json or built-in
        default — see _load_format_config / format.json.example)."""
        fmt = weight_format or self.weight_format
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

    def load_model(self):
        """Load T5 + wav2vec2 + VAE + the default DiT variant ("single").
        Everything except the DiT stays resident for the life of the
        process; see load_dit() for how the DiT itself gets swapped later.
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
        self.weight_format = self.format_config.get(
            "default_weight_format", DEFAULT_WEIGHT_FORMAT)
        defaults = WEIGHT_FORMAT_DEFAULTS.get(
            self.weight_format, WEIGHT_FORMAT_DEFAULTS["fp8_lora"])
        print(f"Weight format: {self.weight_format} "
              f"(default sample_steps={defaults['sample_steps']}, "
              f"text_guidance={defaults['text_guidance']}, "
              f"audio_guidance={defaults['audio_guidance']})")

        base_dir = str(Path(MODEL_PATH) / self.format_config["base_dir"])
        wav2vec_dir = str(Path(MODEL_PATH) / self.format_config["wav2vec_dir"])

        print(f"Loading InfiniteTalk pipeline (base={base_dir}, "
              f"wav2vec={wav2vec_dir}) — this loads T5/VAE/wav2vec2 "
              f"(resident) plus the initial 'single' DiT ...")
        start = time.time()

        PipelineCls = _import_pipeline_class()

        # Best-effort constructor call, modeled on WanI2V's __init__ (device
        # placement, checkpoint_dir) extended with InfiniteTalk-specific
        # args from the plan doc's CLI table (wav2vec_dir, quant_dir). Exact
        # kwarg names are a guess — check generate_infinitetalk.py's own
        # PipelineCls(...) call site once the real repo is available.
        self.pipe = PipelineCls(
            ckpt_dir=base_dir,
            wav2vec_dir=wav2vec_dir,
            quant_dir=self._checkpoint_path("single"),
            device_id=0,
            rank=0,
            t5_cpu=True,
            init_on_cpu=True,
            num_persistent_param_in_dit=self.num_persistent_param_in_dit,
            use_teacache=self.use_teacache,
        )
        self.current_dit_variant = "single"

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
        `.vae` / `.text_encoder` attributes — cosmetic only, doesn't block
        generation."""
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
        """Swap the resident DiT to `variant` ("single" or "multi") if it
        isn't already loaded. T5/wav2vec2/VAE are untouched.

        This is the "option 2" swap from MULTITALK-IMPLEMENTATION.md's
        "Requirement: both single-person and two-person modes" section —
        see the TODO in this file's module docstring for option 1 (test
        whether `multi` alone covers single-person jobs, which would make
        this swap unnecessary).
        """
        import torch

        fmt = weight_format or self.weight_format
        if variant == self.current_dit_variant and fmt == self.weight_format:
            return

        ckpt_path = self._checkpoint_path(variant, fmt)
        print(f"[dit-swap] {self.current_dit_variant!r} -> {variant!r} "
              f"(format={fmt}): loading {ckpt_path}")
        t0 = time.time()

        # Locate the live DiT submodule on the pipeline. Assumed attribute
        # name `.model`, matching Wan2.1's WanI2V convention (a single DiT,
        # no high/low-noise split like Wan2.2's MoE — InfiniteTalk inherits
        # Wan2.1's non-MoE architecture per the plan doc's base-model
        # comparison table). Verify against the real pipeline object in
        # Phase 3; if it's named differently (e.g. `.dit`, `.transformer`),
        # fix the attribute name here.
        dit_module = getattr(self.pipe, "model", None)
        if dit_module is None:
            raise AttributeError(
                "InfiniteTalk pipeline has no '.model' attribute — the "
                "assumed DiT submodule name is wrong. Inspect the real "
                "pipeline object (dir(self.pipe)) in Phase 3 and fix "
                "load_dit()'s `getattr(self.pipe, \"model\", None)` call."
            )

        state_dict = _load_dit_state_dict(ckpt_path)
        missing, unexpected = dit_module.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"    [dit-swap] {len(missing)} missing keys after load "
                  f"(first 5: {missing[:5]})")
        if unexpected:
            print(f"    [dit-swap] {len(unexpected)} unexpected keys after "
                  f"load (first 5: {unexpected[:5]})")
        dit_module.to(self.device)

        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        self.current_dit_variant = variant
        self.weight_format = fmt
        print(f"[dit-swap] done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(self, params: dict) -> dict:
        """Run one generation job. `params` mirrors the request dict built
        by handler_v2.py's send_to_model_server() call.

        Best-effort port of generate_infinitetalk.py's inference path (CLI
        flags per MULTITALK-IMPLEMENTATION.md's "Acceleration" section) as
        a single in-process pipe.generate(...) call, instead of hand-rolling
        a custom diffusion loop the way ~/wan22-14B-fp8-4steps/model_server.py's
        _lightning_generate() does — we have no verified insight into
        InfiniteTalk's DiT forward signature / L-RoPE person-binding
        mechanics, so faithfully reproducing a custom sampling loop for it
        would be pure fabrication. Delegating to the pipeline's own
        `.generate()` (assumed to exist, mirroring WanI2V.generate()) is the
        honest "best-effort port" per the task scope — fix the call
        signature below once the real repo is available.
        """
        image_path = params["image_path"]
        prompt = params.get("prompt", "")
        person_audio_paths = params["person_audio_paths"]  # {"person1": path, "person2": path?}
        audio_type = params.get("audio_type", "para")
        resolution = params.get("resolution", "480p")
        frame_num = params.get("frame_num", 81)
        output_path = params["output_path"]

        variant = "multi" if "person2" in person_audio_paths else "single"

        defaults = WEIGHT_FORMAT_DEFAULTS.get(
            self.weight_format, WEIGHT_FORMAT_DEFAULTS["fp8_lora"])
        sample_steps = params.get("sample_steps", defaults["sample_steps"])
        text_guidance = params.get("text_guidance_scale", defaults["text_guidance"])
        audio_guidance = params.get("audio_guidance_scale", defaults["audio_guidance"])

        if resolution not in RESOLUTIONS:
            raise ValueError(f"Unknown resolution '{resolution}'. Choose 480p or 720p.")
        max_area = RESOLUTIONS[resolution]

        # Swap DiT if this job's person-count doesn't match what's resident.
        self.load_dit(variant)

        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        print(f"Generating {resolution} (variant={variant}), {frame_num} frames, "
              f"{sample_steps} steps, text_guidance={text_guidance}, "
              f"audio_guidance={audio_guidance}, audio_type={audio_type}, "
              f"teacache={self.use_teacache}")

        self._timings = {"t5": 0.0, "wav2vec": 0.0, "vae_encode": 0.0, "vae_decode": 0.0}

        # Best-effort call signature — kwarg names map directly onto the CLI
        # flags documented in MULTITALK-IMPLEMENTATION.md's "Acceleration"
        # section (--sample_steps, --sample_text_guide_scale,
        # --sample_audio_guide_scale, --mode streaming,
        # --num_persistent_param_in_dit, --use_teacache) plus the
        # cond_audio / audio_type fields from the "API Design" section's
        # input_json schema. NOT VERIFIED against the real pipeline class.
        video_tensor = self.pipe.generate(
            input_prompt=prompt,
            img=image,
            cond_audio=person_audio_paths,
            audio_type=audio_type,
            max_area=max_area,
            frame_num=frame_num,
            sample_steps=sample_steps,
            text_guide_scale=text_guidance,
            audio_guide_scale=audio_guidance,
            mode="streaming",
            num_persistent_param_in_dit=self.num_persistent_param_in_dit,
            use_teacache=self.use_teacache,
            teacache_thresh=self.teacache_thresh,
            seed=-1,
        )

        self._write_video(video_tensor, output_path)
        return {"variant": variant, "sample_steps": sample_steps}

    def _write_video(self, video_tensor, output_path: str) -> None:
        """(C, N, H, W) [-1,1] tensor -> mp4. Same conversion as
        ~/wan22-14B-fp8-4steps/model_server.py's generate_video(). fps=25
        per InfiniteTalk's documented duration math (MULTITALK-IMPLEMENTATION.md
        "Handler responsibilities" — NOT independently confirmed, the doc's
        own frame/duration numbers are internally inconsistent; re-derive
        empirically in Phase 3/6 the same way the 113-frame wan22 cap was
        found via OOM testing, not from docs)."""
        import imageio
        import numpy as np

        frames = (video_tensor.clamp(-1, 1) + 1) / 2
        frames = frames.permute(1, 2, 3, 0).cpu().float().numpy()
        frames = (frames * 255).astype(np.uint8)

        writer = imageio.get_writer(output_path, fps=25, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        if not Path(output_path).exists():
            raise RuntimeError(f"Video not created at {output_path}")

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
