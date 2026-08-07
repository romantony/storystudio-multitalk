# InfiniteTalk (audio-driven talking-avatar) RunPod Serverless Dockerfile
#
# Built on MeiGen-AI/InfiniteTalk (successor to MeiGen-MultiTalk, same
# Wan2.1-I2V-14B-480P base). See MULTITALK-IMPLEMENTATION.md's
# "Update 2026-08-05" section for why InfiniteTalk over MultiTalk, and
# ~/wan22-14B-fp8-4steps/ for the reference pattern this Dockerfile mirrors
# (cu128 torch pin, best-effort FlashAttention2 with SDPA fallback).
FROM runpod/pytorch:1.0.7-cu1290-torch260-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/runpod-volume/multitalk \
    HF_HOME=/runpod-volume/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /workspace

# Clone InfiniteTalk's native code — provides the generate_infinitetalk.py
# pipeline (audio-driven I2V on top of Wan2.1-I2V-14B-480P).
#
# NOTE: unlike the wan22-14B-fp8-4steps Dockerfile, we deliberately do NOT
# apply any source patches here (no wan/__init__.py trim, no rope_apply
# float64->float32 downcast, no sed on t5.py device calls). Those patches
# were reverse-engineered against the actual Wan2.2 repo's file layout,
# which we have not verified against InfiniteTalk's repo (not cloned during
# this scaffolding pass — see task boundaries). InfiniteTalk is also
# Wan2.1-based so some of the same float64 RoPE / device-string issues may
# well exist here too — if VRAM or "device=torch.cuda.current_device()"
# issues show up during Phase 3 pod testing, port the same two patches from
# ~/wan22-14B-fp8-4steps/Dockerfile (lines ~30-67, ~28) against the real
# file paths in this clone.
RUN git clone --depth 1 https://github.com/MeiGen-AI/InfiniteTalk.git /workspace/infinitetalk

# Pin torch + torchvision to a CUDA 12.8 build. cu128 supports BOTH Blackwell
# (RTX 5090) and Ada (RTX 6000 Ada / L40S), and runs on any host whose driver
# supports CUDA >= 12.8 (incl. the 12.9 Ada hosts). Installing torchvision
# unpinned would otherwise drag in a cu130 torch that needs a CUDA-13 driver
# many RunPod hosts don't have, crashing torch.cuda init with "driver too old".
RUN python3 -m pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip cache purge

# FlashAttention 2 (best-effort). InfiniteTalk (Wan2.1-derived) routes
# attention to the real flash kernel when `flash_attn` imports; otherwise it
# falls back to PyTorch SDPA. Try prebuilt wheels matching torch 2.7 / cu12 /
# cp310 across both C++ ABIs and a few versions. NON-FATAL: a wheel mismatch
# must not break the build — we confirm activation from the startup log and
# pin the exact wheel later if needed.
#
# 2026-08-06: restricted to 2.7.4.post1 only (was 2.8.2 first) — xformers==
# 0.0.30 (a real, undeclared-upstream dependency, see requirements.txt)
# hard-checks flash_attn.__version__ against the range [2.7.1, 2.7.4] at
# import time (xformers/ops/fmha/flash.py) and raises ImportError otherwise;
# a deploy crashed on "Requires Flash-Attention version >=2.7.1,<=2.7.4 but
# got 2.8.2" once 2.8.2 installed successfully. There IS an env var to
# bypass the check (XFORMERS_IGNORE_FLASH_VERSION_CHECK=1) but skipping it
# risks a real runtime API mismatch, not just an import-time nicety —
# safer to install a version xformers actually declares support for.
# CONFIRMED via the GitHub releases API that v2.7.4.post1 is the ONLY
# release in that range that actually published a torch2.7/cp310 wheel —
# checked v2.7.4, v2.7.3, v2.7.2(.post1), v2.7.1.post4 too and none of them
# have ANY release assets for our torch/python combo (source-only tags),
# so this isn't a preference, it's the sole option that exists.
RUN set +e; \
    for FA in 2.7.4.post1; do \
      for ABI in TRUE FALSE; do \
        URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA}/flash_attn-${FA}+cu12torch2.7cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"; \
        echo "Trying flash-attn $FA abi=$ABI"; \
        python3 -m pip install --no-cache-dir "$URL" && break 2; \
      done; \
    done; \
    python3 -c "import flash_attn" 2>/dev/null || \
      python3 -m pip install --no-cache-dir flash_attn --prefer-binary --no-build-isolation 2>&1 | tail -3; \
    python3 -c "import flash_attn; print('flash-attn', flash_attn.__version__)" \
      || echo "flash-attn NOT installed — runtime will use PyTorch SDPA fallback"; \
    python3 -m pip cache purge; true

# Redirect flash_attention imports to SDPA fallback ONLY when flash_attn is
# unavailable. If flash_attn installed successfully above, native
# flash_attention() uses the FA2 kernel and this step is a no-op. Scoped
# generically across the whole clone (pattern-matches known Wan-family
# import spellings) since we haven't verified InfiniteTalk's exact file
# layout — if none of these patterns match, this step is a harmless no-op
# and SDPA fallback must instead be confirmed/added manually in Phase 3.
RUN python3 - << 'PYEOF'
import sys
try:
    import flash_attn
    print(f"flash_attn {flash_attn.__version__} present — native FA2 kernel active, skipping redirect")
    sys.exit(0)
except ImportError:
    pass
import pathlib
subs = [
    ('from .attention import flash_attention',            'from .attention import attention as flash_attention'),
    ('from ..modules.attention import flash_attention',   'from ..modules.attention import attention as flash_attention'),
    ('from wan.modules.attention import flash_attention', 'from wan.modules.attention import attention as flash_attention'),
]
patched = 0
for py in pathlib.Path('/workspace/infinitetalk').rglob('*.py'):
    try:
        code = py.read_text()
    except Exception:
        continue
    updated = code
    for old, new in subs:
        updated = updated.replace(old, new)
    if updated != code:
        py.write_text(updated)
        print(f"  patched {py}")
        patched += 1
print(f"flash_attn unavailable — redirected flash_attention→attention() in {patched} file(s) (SDPA fallback)")
PYEOF

# Rewrite deprecated torch.cuda.amp.autocast()/amp.autocast() call sites to
# the non-deprecated torch.amp.autocast('cuda', ...) form. Purely cosmetic
# (both APIs are functionally identical — torch.cuda.amp.autocast is just a
# 'cuda'-pinned wrapper around torch.amp.autocast) but the FutureWarning
# these throw on every autocast entry was flooding worker logs badly enough
# to obscure real signal (e.g. actual step-timing/error lines) during
# debugging. Two ordered passes since a plain global substitution would
# double-patch: pass 1 handles the fully-qualified `torch.cuda.amp.autocast(`
# form (clip.py only); pass 2 handles the bare `amp.autocast(` form used
# everywhere else via `import torch.cuda.amp as amp`, with a negative
# lookbehind so it skips the `torch.amp.autocast(` text pass 1 just wrote
# (that new text also contains the substring `amp.autocast(`, which would
# otherwise get matched again as a false positive and mangled into
# `torch.torch.amp.autocast(...)`). Verified against a scratch clone of this
# same repo — all 10 affected files (model.py, multitalk_model.py, vae.py,
# clip.py, vace.py/vace_model.py, image2video.py, text2video.py,
# first_last_frame2video.py, xdit_context_parallel.py) py_compile clean
# after patching, decorator form (@amp.autocast(enabled=False)) and
# multi-line call form both handled correctly.
RUN python3 - << 'PYEOF'
import pathlib
import re

qualified_re = re.compile(r'torch\.cuda\.amp\.autocast\(')
bare_re = re.compile(r'(?<!torch\.)amp\.autocast\(')

patched = 0
for py in pathlib.Path('/workspace/infinitetalk').rglob('*.py'):
    try:
        code = py.read_text()
    except Exception:
        continue
    updated = qualified_re.sub("torch.amp.autocast('cuda', ", code)
    updated = bare_re.sub("torch.amp.autocast('cuda', ", updated)
    if updated != code:
        py.write_text(updated)
        print(f"  patched {py}")
        patched += 1
print(f"Rewrote deprecated autocast() calls -> torch.amp.autocast('cuda', ...) in {patched} file(s)")
PYEOF

# Remove @torch.compile from calculate_x_ref_attn_map (wan/utils/
# multitalk_utils.py, the only torch.compile use in the whole repo — grepped
# to confirm). This function builds the ref_target_masks person-targeting
# attention map: a Python-level `for class_idx in ...` loop containing a
# torch_gc() (= torch.cuda.empty_cache() + torch.cuda.ipc_collect(), both
# GPU-synchronizing) call, wrapping tensor ops whose shapes can vary between
# calls. That combination — dynamic control flow + host-device sync inside a
# compiled region — is a known torch.compile/Dynamo recompilation trap:
# every shape/guard miss forces a full Triton/Inductor recompile, which can
# cost tens of seconds on its own. Measured step-timing (added separately in
# model_server.py) showed ~280-300s/step even after confirming the DiT is
# fully VRAM-resident and flash-attention is active everywhere else — this
# function, called once per self-attention layer (~40 blocks) per forward
# pass (up to 4 passes/step), is the next strongest suspect. This patch
# removes ONLY the decorator; the function runs the identical math eager
# instead of compiled, so it isolates torch.compile's own overhead as an
# experiment rather than changing any generation behavior.
RUN python3 - << 'PYEOF'
import pathlib
p = pathlib.Path('/workspace/infinitetalk/wan/utils/multitalk_utils.py')
code = p.read_text()
old = "@torch.compile\ndef calculate_x_ref_attn_map("
new = "def calculate_x_ref_attn_map("
count = code.count(old)
if count == 1:
    p.write_text(code.replace(old, new))
    print("removed @torch.compile from calculate_x_ref_attn_map")
elif count == 0:
    print("WARNING: @torch.compile decorator pattern not found — "
          "upstream source may have changed, patch is now a no-op, "
          "check manually")
else:
    raise RuntimeError(f"expected exactly 1 occurrence, found {count} — "
                        f"refusing to patch ambiguously")
PYEOF

# Core deps + audio-conditioning deps + InfiniteTalk's own direct
# dependencies (xfuser, optimum-quanto, etc.) — installed from
# requirements.txt, the single source of truth. NOTE: this used to be a
# second, hand-maintained package list duplicated here in the Dockerfile,
# which silently drifted out of sync with requirements.txt (a live deploy
# crashed on missing xfuser/pyloudnorm/optimum-quanto etc. that had already
# been added to requirements.txt but never to this list) — fixed 2026-08-06
# by deleting the duplicate and installing from the file directly.
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip cache purge

# Verify critical packages import and confirm the torch CUDA build is 12.x (not 13.x).
# Also print torch's C++ ABI + flash-attn status so we can pin the right wheel if missed.
RUN python3 -c "import runpod, diffusers, torch, torchvision, easydict, librosa, soundfile; \
print(f'OK — runpod={runpod.__version__} diffusers={diffusers.__version__} torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda}'); \
print(f'torch cxx11_abi={torch._C._GLIBCXX_USE_CXX11_ABI}'); \
assert torch.version.cuda.startswith('12'), f'torch CUDA build {torch.version.cuda} requires too-new a driver'" && \
    (python3 -c "import flash_attn; print('flash-attn', flash_attn.__version__, 'OK')" \
     || echo "flash-attn not present — SDPA fallback")

# Verify InfiniteTalk source was cloned correctly. Only checks the top-level
# entrypoint script — we haven't verified the internal package layout
# (wan/ subpackage names etc.), see model_server.py's TODOs for the
# assumptions made about importable module paths.
RUN test -f /workspace/infinitetalk/generate_infinitetalk.py && \
    echo "InfiniteTalk source OK"

COPY handler_v2.py ./handler.py
COPY model_server.py ./handler/model_server.py

RUN mkdir -p /workspace/models /workspace/huggingface /workspace/outputs /workspace/handler

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "print('healthy')" || exit 1

CMD ["python3", "-u", "handler.py"]
