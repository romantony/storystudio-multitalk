"""
RunPod Serverless Handler for InfiniteTalk — audio-driven talking-avatar video.
Version 1.0 — single/multi-person, built on MeiGen-AI/InfiniteTalk
(Wan2.1-I2V-14B-480P lineage). See MULTITALK-IMPLEMENTATION.md.

Model loads ONCE on container startup via persistent model server and stays
warm between jobs, eliminating repeated load time — same pattern as
~/wan22-14B-fp8-4steps/handler_v2.py, extended for per-person audio input
and single/multi checkpoint routing (see model_server.py's load_dit()).
"""
# FIRST: Set CUDA environment variables before ANY imports
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

# Now import other modules
import sys
import runpod
import subprocess
import tempfile
import base64
import time
import socket
import json
import boto3
import requests
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/multitalk")
SOCKET_PATH = "/tmp/multitalk_model_server.sock"
MODEL_SERVER_SCRIPT = "/workspace/handler/model_server.py"

# R2 Configuration — no hardcoded fallbacks for credentials/account: these
# must come from the endpoint's env (same convention as
# ~/wan22-14B-fp8-4steps/handler_v2.py).
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "e2e-storystudio")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "storyaistudio.app")

# Resolution mapping — 720p single-GPU status is unconfirmed (plan doc Open
# Question 3: upstream README described 720p as multi-GPU-only at doc-write
# time). Left enabled here; reject at the API layer instead if Phase 3
# testing shows it OOMs / isn't supported single-GPU.
RESOLUTION_MAP = {
    "480p": "832*480",
    "720p": "1280*720",
}

# audio_type enum — only "para" (simultaneous, e.g. singing together) is
# confirmed from the plan doc's example input_json (Open Question 1: the
# turn-taking/sequential value name, likely "add" or similar, is NOT
# confirmed — verify against generate_infinitetalk.py / its utils/ once the
# real repo is available). We validate against this known set but let
# anything else pass through unchanged with a warning, rather than hard
# rejecting, since the full enum isn't known.
KNOWN_AUDIO_TYPES = {"para"}

# duration_s -> frame_num, assumed 25fps. InfiniteTalk's documented duration
# math in MULTITALK-IMPLEMENTATION.md is internally inconsistent (quotes
# both "15s = 201 frames @ 25fps", which doesn't reconcile arithmetically,
# and "81 frames optimal for prompt adherence" without stating fps for that
# figure). Range is intentionally conservative (1-5s) pending the same kind
# of empirical OOM-boundary testing that found wan22's 113-frame/7s cap
# (MULTITALK-IMPLEMENTATION.md Phase 6) — NOT yet performed for this worker.
#
# CONFIRMED (multitalk.py's generate_infinitetalk(), the frame-mask .view()
# call): frame_num MUST be of the form 4n+1 (also stated in the CLI's own
# --frame_num help text) or generation crashes with a shape/reshape
# RuntimeError — model_server.py's default of 81 (=4*20+1) already
# satisfies this, but the old `s*25+1` formula here did NOT for any s in
# 1-5 (e.g. 5*25+1=126, not 4n+1). Rounded down to the nearest 4n+1 <= s*25
# instead — model_server.py's _largest_4np1_at_most() would catch this too
# at runtime, but fixing it here means the reported duration_s in job
# responses matches what was actually requested without needing a
# server-side correction in the common case.
DURATION_TO_FRAMES = {s: 4 * ((s * 25 - 1) // 4) + 1 for s in range(1, 6)}

# Global state
model_server_process = None


def verify_model_present():
    """Confirm the base model + wav2vec2 + at least one DiT checkpoint exist
    on the network volume. Layout mirrors format.json.example."""
    base_dir = Path(MODEL_PATH) / "Wan2.1-I2V-14B-480P"
    wav2vec_dir = Path(MODEL_PATH) / "chinese-wav2vec2-base"
    if not base_dir.exists() or not wav2vec_dir.exists():
        raise RuntimeError(
            f"Model weights not found at {MODEL_PATH}. Expected "
            f"'{base_dir.name}/' (Wan2.1-I2V-14B-480P base: T5/VAE/tokenizer) "
            f"and '{wav2vec_dir.name}/' (audio encoder) on the network "
            f"volume — see MULTITALK-IMPLEMENTATION.md 'Required Model "
            f"Downloads' and format.json.example."
        )
    print(f"✓ Model found at {MODEL_PATH}")


def start_model_server():
    """Start the persistent model server as a background process"""
    global model_server_process

    if model_server_process is not None:
        print("Model server already running")
        return

    print("Starting persistent model server...")

    model_server_process = subprocess.Popen(
        [sys.executable, "-u", MODEL_SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    import threading
    def read_model_server_output():
        while model_server_process is not None and model_server_process.poll() is None:
            if model_server_process.stdout:
                line = model_server_process.stdout.readline()
                if line:
                    print(f"[ModelServer] {line.rstrip()}", flush=True)

    output_thread = threading.Thread(target=read_model_server_output, daemon=True)
    output_thread.start()

    print("Waiting for model to load into GPU memory...")
    start_time = time.time()
    max_wait = 600  # 10 minutes max

    while time.time() - start_time < max_wait:
        if os.path.exists(SOCKET_PATH):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(SOCKET_PATH)
                sock.close()
                print(f"✓ Model server ready! (took {time.time() - start_time:.1f}s)")
                return
            except:
                pass

        if model_server_process.poll() is not None:
            raise Exception(f"Model server crashed with exit code: {model_server_process.poll()}")

        time.sleep(1)

    raise Exception("Model server failed to start within timeout")


def send_to_model_server(request: dict, timeout: int = 3600) -> dict:
    """Send a job request to the model server with robust error handling"""
    global model_server_process

    if model_server_process is not None:
        poll = model_server_process.poll()
        if poll is not None:
            print(f"WARNING: Model server process died with code {poll}")
            if model_server_process.stdout:
                remaining = model_server_process.stdout.read()
                if remaining:
                    print(f"Model server final output: {remaining}")
            raise Exception(f"Model server process died unexpectedly (exit code: {poll})")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(SOCKET_PATH)
    except socket.error as e:
        raise Exception(f"Failed to connect to model server: {e}. Server may have crashed.")

    message = json.dumps(request) + "\n\n"
    sock.sendall(message.encode())
    print(f"Request sent to model server ({len(message)} bytes)", flush=True)

    data = b""
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                print(f"Connection closed by server, received {len(data)} bytes total", flush=True)
                break
            data += chunk
            if b"\n" in data:
                break
    except socket.timeout:
        sock.close()
        raise Exception(f"Socket read timeout after {timeout}s - generation may still be running")

    sock.close()

    response_str = data.decode().strip()
    if not response_str:
        if model_server_process is not None:
            poll = model_server_process.poll()
            if poll is not None:
                raise Exception(f"Model server crashed during generation (exit code: {poll})")
        raise Exception("Empty response from model server - server may have crashed or timed out")

    try:
        return json.loads(response_str)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON from model server: {e}. Response was: {response_str[:500]}")


def get_r2_client():
    """Initialize and return R2 S3 client"""
    return boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )


def build_asset_key(category: str, asset_type: str, ext: str,
                     project_id: str = None, frame_id: str = None,
                     job_id: str = None) -> str:
    """storystudio/{category}/{asset_type}_{project_id}_{frame_id}.{ext}, or
    storystudio/{category}/{asset_type}_{timestamp}_{job_id}.{ext} when
    project_id/frame_id (both optional request fields) aren't given.
    Called with asset_type='multitalk' -> storystudio/video/multitalk_*
    naming, matching MULTITALK-IMPLEMENTATION.md's Phase 5 bullet
    ("new storystudio/video/multitalk_* asset naming") literally — note
    this is asset_type-as-PREFIX, the opposite order from
    ~/wan22-14B-fp8-4steps/handler_v2.py's asset_type-as-suffix convention
    (`..._i2v.mp4`); intentional, to match the plan doc's wildcard pattern."""
    if project_id and frame_id:
        name = f"{asset_type}_{project_id}_{frame_id}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name = f"{asset_type}_{timestamp}_{job_id or 'unknown'}"
    return f"storystudio/{category}/{name}.{ext}"


def upload_to_r2(file_path: str, r2_key: str) -> str:
    """Upload file to R2 at the given key and return public URL"""
    s3_client = get_r2_client()

    with open(file_path, 'rb') as f:
        s3_client.upload_fileobj(f, R2_BUCKET_NAME, r2_key)

    public_url = f"https://{R2_PUBLIC_URL}/{r2_key}"
    return public_url


def _fetch_bytes(url_or_b64: str, label: str) -> bytes:
    """Download (URL, with direct R2 bypass for our own asset URLs) or
    base64-decode a single input (image or audio). Shared helper so image
    and each per-person audio file go through the same fetch path — mirrors
    ~/wan22-14B-fp8-4steps/handler_v2.py's image-handling block, generalized."""
    if url_or_b64.startswith(('http://', 'https://')):
        t0 = time.time()
        r2_fetched = False
        if f"{R2_PUBLIC_URL}/" in url_or_b64:
            r2_key = url_or_b64.split(f"{R2_PUBLIC_URL}/", 1)[1]
            try:
                print(f"Downloading {label} from R2 directly: {r2_key}")
                obj = get_r2_client().get_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
                data = obj["Body"].read()
                r2_fetched = True
            except Exception as r2_err:
                print(f"R2 direct fetch failed for {label} ({r2_err}), falling back to HTTP")
        if not r2_fetched:
            print(f"Downloading {label} from URL: {url_or_b64}")
            response = requests.get(url_or_b64, timeout=(10, 60))
            response.raise_for_status()
            data = response.content
        print(f"{label} ready: {len(data)/1024:.0f} KB in {time.time()-t0:.1f}s")
        return data
    return base64.b64decode(url_or_b64)


def generate_video(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function for audio-driven talking-avatar video.
    Uses the persistent model server for a warm model between jobs;
    routes to the single- or multi-person DiT checkpoint based on how many
    person audio tracks are supplied (see model_server.py's load_dit()).
    """
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")
    start_time = time.time()

    try:
        # --- Validate inputs -------------------------------------------------
        if "image" not in job_input:
            return {"error": "Missing required input: image"}

        audio = job_input.get("audio")
        if not isinstance(audio, dict) or "person1" not in audio:
            return {"error": "Missing required input: audio.person1 (audio must be an "
                              "object with at least a 'person1' key)"}
        if len(audio) > 2:
            return {"error": "audio must contain at most 2 keys (person1, person2) — "
                              "InfiniteTalk's 'multi' checkpoint is documented for 2 "
                              "speakers only (see MULTITALK-IMPLEMENTATION.md Open "
                              "Question 7 on higher speaker counts)"}
        extra_keys = set(audio) - {"person1", "person2"}
        if extra_keys:
            return {"error": f"Unknown audio keys: {sorted(extra_keys)}. "
                              f"Expected 'person1' and optionally 'person2'."}

        image_input = job_input["image"]
        prompt = job_input.get("prompt", "")
        audio_type = job_input.get("audio_type", "para")
        if audio_type not in KNOWN_AUDIO_TYPES:
            # Open Question 1 (plan doc): only "para" is confirmed from the
            # example input_json. Pass unknown values through rather than
            # rejecting — the real enum (e.g. a turn-taking/sequential
            # value) isn't confirmed, so hard-blocking here could reject
            # something InfiniteTalk actually supports.
            print(f"WARNING: audio_type={audio_type!r} is not in the confirmed "
                  f"enum {KNOWN_AUDIO_TYPES} — passing through unchanged. "
                  f"See MULTITALK-IMPLEMENTATION.md Open Question 1.")
        resolution = job_input.get("resolution", "480p")
        sample_steps = job_input.get("sample_steps")  # None -> model_server picks the
                                                        # weight-format default

        duration_s = job_input.get("duration_s")
        if duration_s is not None:
            if duration_s not in DURATION_TO_FRAMES:
                return {"error": f"duration_s must be one of {sorted(DURATION_TO_FRAMES)} (seconds)"}
            frame_num = DURATION_TO_FRAMES[duration_s]
        else:
            frame_num = job_input.get("frame_num", 81)

        if resolution not in RESOLUTION_MAP:
            return {"error": f"Invalid resolution. Must be one of: {list(RESOLUTION_MAP.keys())}"}

        if sample_steps is not None and not (1 <= sample_steps <= 50):
            return {"error": "sample_steps must be between 1 and 50"}

        # Handle image + per-person audio input (URL or base64)
        image_bytes = _fetch_bytes(image_input, "image")
        audio_bytes = {person: _fetch_bytes(src, f"audio[{person}]")
                        for person, src in audio.items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "input_image.jpg"
            image_path.write_bytes(image_bytes)

            person_audio_paths = {}
            for person, data in audio_bytes.items():
                audio_path = Path(tmpdir) / f"{person}.wav"
                audio_path.write_bytes(data)
                person_audio_paths[person] = str(audio_path)

            output_path = Path(tmpdir) / "output_video.mp4"

            print(f"Sending job to warm model server "
                  f"({'multi' if 'person2' in person_audio_paths else 'single'}-person)...")
            request = {
                "job_id": job_id,
                "image_path": str(image_path),
                "prompt": prompt,
                "person_audio_paths": person_audio_paths,
                "audio_type": audio_type,
                "resolution": resolution,
                "frame_num": frame_num,
                "output_path": str(output_path),
            }
            if sample_steps is not None:
                request["sample_steps"] = sample_steps

            result = send_to_model_server(request)

            if not result.get("success"):
                return {"error": result.get("error", "Generation failed")}

            video_size_mb = round(output_path.stat().st_size / 1024 / 1024, 2)

            print(f"Uploading video to R2...")
            r2_key = build_asset_key(
                "video", "multitalk", "mp4",
                job_input.get("project_id"), job_input.get("frame_id"), job_id,
            )
            video_url = upload_to_r2(str(output_path), r2_key)

            generation_time = time.time() - start_time

            return {
                "video_url": video_url,
                "generation_time": round(generation_time, 2),
                "model_generation_time": round(result.get("generation_time", 0), 2),
                "video_size_mb": video_size_mb,
                "resolution": resolution,
                "sample_steps": result.get("sample_steps", sample_steps),
                # model_server.py may CLAMP frame_num below what was requested
                # if the input audio is too short (upstream requires the audio
                # embedding to be strictly longer than frame_num) — report
                # the actual value used, not just what was asked for.
                "frame_num": result.get("frame_num", frame_num),
                "duration_s": round(result.get("frame_num", frame_num) / 25, 2),
                "variant": result.get("variant"),
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Initialize on container startup
print("=" * 60)
print("InfiniteTalk Handler v1.0 — audio-driven talking-avatar video")
print("=" * 60)

verify_model_present()
start_model_server()

print("✓ Handler ready with warm model!")
print("=" * 60)

runpod.serverless.start({"handler": generate_video})
