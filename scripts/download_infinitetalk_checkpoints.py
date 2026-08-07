#!/usr/bin/env python3
"""
Download InfiniteTalk's own weights (audio-conditioned DiT checkpoints +
quantized T5) from MeiGen-AI/InfiniteTalk onto the network volume, in the
layout model_server.py / format.json.example expect.

Run this on a RunPod CPU/storage pod with the target network volume mounted.

IMPORTANT: interactive/storage Pods often mount the network volume at
/workspace, NOT /runpod-volume (/runpod-volume there is just the pod's local
ephemeral disk and gets wiped when the pod is deleted). The Serverless
endpoint that actually runs this worker mounts the same volume at
/runpod-volume — so as long as you download to the volume's real mount
point on this pod, it will show up at /runpod-volume/multitalk for the
deployed worker. Check `df -h` / `mount` on your pod to confirm where the
network volume is actually mounted before running this.

    pip install -U huggingface_hub
    python3 scripts/download_infinitetalk_checkpoints.py --dest /workspace/multitalk

File listing below is VERIFIED 2026-08-06 against the live HF repo tree
(huggingface.co/api/models/MeiGen-AI/InfiniteTalk) — NOT the same as
MULTITALK-IMPLEMENTATION.md's original size table, which assumed a
`infinitetalk_single_fp8_lora.safetensors` file that does not actually
exist upstream. Only `multi` ships a `_lora` fp8 variant; `single` only has
plain `fp8` + `int8`/`int8_lora`. Every quant .safetensors file also ships
a same-named sidecar .json (scale/mapping metadata) — this script fetches
those too.

This does NOT fetch the Wan2.1-I2V-14B-480P base (T5/VAE/CLIP/tokenizer) or
chinese-wav2vec2-base — see download_base_encoders.py for those.
"""
import argparse
import json
import os
from pathlib import Path

REPO_ID = "MeiGen-AI/InfiniteTalk"

# (relative HF path, human label) — the "core" quant set actually usable by
# model_server.py's default per-variant weight formats (single->fp8,
# multi->fp8_lora), plus their sidecar .json metadata files and the shared
# t5_fp8 files. This is the lean/production set (~47 GB) and is fetched by
# default.
CORE_QUANT_FILES = [
    "quant_models/infinitetalk_single_fp8.safetensors",
    "quant_models/infinitetalk_single_fp8.json",
    "quant_models/infinitetalk_multi_fp8_lora.safetensors",
    "quant_models/infinitetalk_multi_fp8_lora.json",
    "quant_models/t5_fp8.safetensors",
    "quant_models/t5_map_fp8.json",
    "quant_models/quant.json",
]

# Everything else in quant_models/ — the non-default fp8 (multi, no lora)
# and all int8/int8_lora variants + their sidecars. Only fetched with
# --all-quant-variants; useful for the Phase 7 "INT8 quality delta" check
# and for comparing multi_fp8 vs multi_fp8_lora, but roughly doubles the
# download (~+70 GB).
EXTRA_QUANT_FILES = [
    "quant_models/infinitetalk_multi_fp8.safetensors",
    "quant_models/infinitetalk_multi_fp8.json",
    "quant_models/infinitetalk_single_int8.safetensors",
    "quant_models/infinitetalk_single_int8.json",
    "quant_models/infinitetalk_single_int8_lora.safetensors",
    "quant_models/infinitetalk_single_int8_lora.json",
    "quant_models/infinitetalk_multi_int8.safetensors",
    "quant_models/infinitetalk_multi_int8.json",
    "quant_models/infinitetalk_multi_int8_lora.safetensors",
    "quant_models/infinitetalk_multi_int8_lora.json",
]

# Full-precision bf16 DiTs — only needed for the Phase 3 "confirm the base
# pipeline works before any of our modifications" baseline test. ~28 GB
# combined (single + multi), not needed for a quant-only production deploy.
BF16_FILES = [
    "single/infinitetalk.safetensors",
    "multi/infinitetalk.safetensors",
]

# FusioniX distillation LoRA — a DIFFERENT HF repo from InfiniteTalk's own
# (REPO_ID above), so it needs its own hf_hub_download() call, not just
# another entry in one of the file lists above. Cuts sample_steps 40->8 on
# top of the bf16 DiT (model_server.py's FUSIONX_LORA_PATH; only applies
# with quant=None, i.e. requires --include-bf16 too — see that flag's
# --help). VERIFIED 2026-08-07 against the live HF API
# (huggingface.co/api/models/vrgamedevgirl84/Wan14BT2VFusioniX) — this is
# the I2V variant, matching InfiniteTalk's I2V-based pipeline (the repo
# also ships T2V/Phantom LoRAs under the same folder, not what we want).
FUSIONX_LORA_REPO_ID = "vrgamedevgirl84/Wan14BT2VFusioniX"
FUSIONX_LORA_FILE = "FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"

# Copied verbatim from format.json.example — keep the two in sync.
FORMAT_JSON = {
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
    "default_weight_format": {"single": "fp8", "multi": "fp8_lora"},
}


def download(dest: Path, files: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(files)} files from {REPO_ID} -> {dest}")

    for f in files:
        target = dest / f
        if target.exists():
            print(f"  skip (already present): {f}")
            continue
        print(f"  fetching {f} ...")
        hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=str(dest))

    format_json = dest / "format.json"
    if not format_json.exists():
        format_json.write_text(json.dumps(FORMAT_JSON, indent=2) + "\n")
        print(f"  wrote {format_json}")
    else:
        print(f"  {format_json} already exists — left untouched")

    print("Done. Verify:")
    for f in files:
        p = dest / f
        size_mb = p.stat().st_size / 1024**2 if p.exists() else 0
        print(f"  {'OK ' if p.exists() else 'MISSING'} {f} ({size_mb:.1f} MB)")


def download_fusionx_lora(dest: Path) -> None:
    """Fetches FUSIONX_LORA_FILE from FUSIONX_LORA_REPO_ID (a different HF
    repo than InfiniteTalk's own) into <dest>/loras/, flattening the
    upstream FusionX_LoRa/ subfolder since that's the only file we want
    from that repo. model_server.py's FUSIONX_LORA_PATH should point at the
    resulting <dest>/loras/Wan2.1_I2V_14B_FusionX_LoRA.safetensors."""
    from huggingface_hub import hf_hub_download

    lora_dest = dest / "loras"
    lora_dest.mkdir(parents=True, exist_ok=True)
    target_name = Path(FUSIONX_LORA_FILE).name
    target = lora_dest / target_name

    if target.exists():
        print(f"  skip (already present): loras/{target_name}")
        return

    print(f"Downloading {FUSIONX_LORA_FILE} from {FUSIONX_LORA_REPO_ID} -> {lora_dest}")
    fetched = hf_hub_download(repo_id=FUSIONX_LORA_REPO_ID, filename=FUSIONX_LORA_FILE)
    import shutil
    shutil.copy(fetched, target)

    size_mb = target.stat().st_size / 1024**2 if target.exists() else 0
    print(f"  {'OK ' if target.exists() else 'MISSING'} loras/{target_name} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest",
        default=os.getenv("DOWNLOAD_DEST", "/workspace/multitalk"),
        help="Destination directory at the network volume's actual mount point on this pod "
             "(default: $DOWNLOAD_DEST or /workspace/multitalk — NOT /runpod-volume, "
             "which is local pod disk on most interactive Pods). Verify with `df -h` first.",
    )
    parser.add_argument(
        "--all-quant-variants", action="store_true",
        help="Also fetch multi_fp8 (non-lora) and all int8/int8_lora variants + sidecars "
             "(~+70 GB) — only needed for Phase 7's INT8 quality-delta comparison.",
    )
    parser.add_argument(
        "--include-bf16", action="store_true",
        help="Also fetch the full-precision single/ and multi/ bf16 DiTs (~28 GB) — "
             "needed for Phase 3's baseline correctness test before trusting the quant path, "
             "and a prerequisite for --fusionx-lora (LoRA only applies on the bf16 DiT).",
    )
    parser.add_argument(
        "--fusionx-lora", action="store_true",
        help="Also fetch the FusioniX distillation LoRA (~a few hundred MB, from a "
             "different HF repo) into <dest>/loras/ — for model_server.py's "
             "FUSIONX_LORA_PATH. Requires --include-bf16 too (and "
             "scripts/download_base_encoders.py --include-dit-shards on the same volume) "
             "since the LoRA only applies on the unquantized bf16 DiT.",
    )
    args = parser.parse_args()

    files = list(CORE_QUANT_FILES)
    if args.all_quant_variants:
        files += EXTRA_QUANT_FILES
    if args.include_bf16:
        files += BF16_FILES

    dest = Path(args.dest)
    download(dest, files)

    if args.fusionx_lora:
        download_fusionx_lora(dest)


if __name__ == "__main__":
    main()
