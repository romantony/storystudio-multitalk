#!/usr/bin/env python3
"""
Download the Wan2.1-I2V-14B-480P base model's encoders (VAE, T5, CLIP,
tokenizers) + the chinese-wav2vec2-base audio encoder onto the network
volume, alongside InfiniteTalk's own weights fetched by
download_infinitetalk_checkpoints.py.

Run this on a RunPod CPU/storage pod with the target network volume
mounted — see download_infinitetalk_checkpoints.py's docstring for the
/workspace vs /runpod-volume mount-point gotcha, same caveat applies here.

    pip install -U huggingface_hub
    python3 scripts/download_base_encoders.py --dest /workspace/multitalk

Note: WanI2V (and by inheritance InfiniteTalk, same Wan2.1 lineage per
MULTITALK-IMPLEMENTATION.md) loads the T5 tokenizer via
  AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "google/umt5-xxl"))
i.e. as a LOCAL path, not an HF repo id — so the google/umt5-xxl/ folder
must physically exist under base_dir, not just the two big checkpoint
files. Same applies to the CLIP tokenizer under xlm-roberta-large/.

Does NOT fetch InfiniteTalk's own DiT/T5-fp8 weights — see
download_infinitetalk_checkpoints.py for those.
"""
import argparse
import os
from pathlib import Path

BASE_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAV2VEC_REPO_ID = "TencentGameMate/chinese-wav2vec2-base"

# VAE + T5 (required — matches wan22-14B-fp8-4steps' download_shared_encoders.py
# pattern, verified 2026-08-06 to exist under these exact names in the
# Wan2.1-I2V-14B-480P repo via huggingface.co/api/models/Wan-AI/Wan2.1-I2V-14B-480P).
CORE_BASE_FILES = [
    "Wan2.1_VAE.pth",                          # ~508 MB
    "models_t5_umt5-xxl-enc-bf16.pth",         # ~11.4 GB
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    # CONFIRMED (read wan/multitalk.py's InfiniteTalkPipeline.__init__
    # directly): the WanModel architecture config, read from
    # `checkpoint_dir/config.json` in ALL THREE DiT-loading branches (quant /
    # bf16-merge / pre-merged dit_path). Small file, no flag needed.
    "config.json",
]

# CLIP image encoder + its tokenizer. Stock Wan2.1-I2V-14B-480P uses this
# for image conditioning (unlike Wan2.2's A14B, whose download script does
# NOT fetch CLIP — the newer arch appears to drop the CLIP dependency).
# Fetched by default here since the file exists in the base repo and I2V
# conditioning conventionally needs it, but this is UNVERIFIED for
# InfiniteTalk specifically: InfiniteTalk's own image conditioning path
# might not route through CLIP at all (audio cross-attn + L-RoPE could
# supersede it). Cheap relative to the DiT (~2.6 GB) so downloaded by
# default rather than risking a second round-trip; skip with --no-clip if
# Phase 3 confirms it's unused.
CLIP_FILES = [
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",  # ~2.6 GB
    "xlm-roberta-large/sentencepiece.bpe.model",
    "xlm-roberta-large/special_tokens_map.json",
    "xlm-roberta-large/tokenizer.json",
    "xlm-roberta-large/tokenizer_config.json",
]

# Base Wan2.1-I2V-14B-480P bf16 DiT shards — required for the FusioniX LoRA
# path (model_server.py's FUSIONX_LORA_PATH), which only applies on top of
# the unquantized bf16 DiT (upstream only merges a LoRA when quant is None
# — see wan/multitalk.py ~line 239). _load_bf16_dit_module in
# model_server.py already expects these under exactly this filename
# pattern; they just were never fetched by this script before (flagged as
# a known gap in that function's docstring). VERIFIED 2026-08-07 against
# the live HF API (huggingface.co/api/models/Wan-AI/Wan2.1-I2V-14B-480P).
# Skipped by default (~16.4 GB, only needed for a FusioniX deploy) — pass
# --include-dit-shards to fetch them.
DIT_SHARD_FILES = [
    f"diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)
] + ["diffusion_pytorch_model.safetensors.index.json"]


def download_base(dest: Path, files: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading base encoders from {BASE_REPO_ID} -> {dest}")

    for f in files:
        target = dest / f
        if target.exists():
            print(f"  skip (already present): {f}")
            continue
        print(f"  fetching {f} ...")
        hf_hub_download(repo_id=BASE_REPO_ID, filename=f, local_dir=str(dest))


def download_wav2vec(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    wav2vec_dest = dest.parent / "chinese-wav2vec2-base"
    wav2vec_dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {WAV2VEC_REPO_ID} -> {wav2vec_dest}")
    snapshot_download(repo_id=WAV2VEC_REPO_ID, local_dir=str(wav2vec_dest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest",
        default=os.getenv("DOWNLOAD_DEST", "/workspace/multitalk"),
        help="Root multitalk/ dir at the network volume's actual mount point on this pod "
             "(default: $DOWNLOAD_DEST or /workspace/multitalk). Base encoder files land "
             "under <dest>/Wan2.1-I2V-14B-480P/, wav2vec2 under <dest>/chinese-wav2vec2-base/ "
             "— same root as download_infinitetalk_checkpoints.py's --dest.",
    )
    parser.add_argument(
        "--no-clip", action="store_true",
        help="Skip the CLIP image encoder (~2.6 GB) — only if Phase 3 confirms "
             "InfiniteTalk doesn't use it (see CLIP_FILES comment above).",
    )
    parser.add_argument(
        "--skip-wav2vec", action="store_true",
        help="Skip chinese-wav2vec2-base (e.g. if already present from a prior run).",
    )
    parser.add_argument(
        "--include-dit-shards", action="store_true",
        help="Also fetch the base Wan2.1 bf16 DiT shards (~16.4 GB) — only needed "
             "for a FusioniX LoRA deploy (model_server.py's FUSIONX_LORA_PATH). "
             "Skipped by default since the standard fp8/int8-quant deploy never "
             "loads these.",
    )
    args = parser.parse_args()

    base_dest = Path(args.dest) / "Wan2.1-I2V-14B-480P"
    files = list(CORE_BASE_FILES)
    if not args.no_clip:
        files += CLIP_FILES
    if args.include_dit_shards:
        files += DIT_SHARD_FILES
    download_base(base_dest, files)

    if not args.skip_wav2vec:
        download_wav2vec(base_dest)

    print("Done. Verify:")
    for f in files:
        p = base_dest / f
        size_mb = p.stat().st_size / 1024**2 if p.exists() else 0
        print(f"  {'OK ' if p.exists() else 'MISSING'} {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
