# ============================================================
# Serve MedGemma IT with vLLM, behind a Cloudflare tunnel.
#
# Run this as one Colab cell. Text only: the vision tower is
# not loaded. It prints the public URL and a fresh API key,
# and exports both so a judge cell in the same notebook needs
# no arguments about the server.
# ============================================================

import os
import re
import sys
import time
import getpass
import secrets
import subprocess
import urllib.request


# ============================================================
# CONFIG
# ============================================================

# MODEL = "google/medgemma-4b-it"
MODEL = "google/medgemma-27b-it"
# SERVED_MODEL = "medgemma-4b-it"
SERVED_MODEL = "medgemma-27b-it"

PORT = 8000

MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 16

# Set to False once you have confirmed the model loads.
# Eager mode disables CUDA graphs and is meaningfully slower
# across tens of thousands of requests.
ENFORCE_EAGER = True


# ============================================================
# 1. INPUTS
# ============================================================

# Colab Secrets first (key icon in the sidebar: add HF_TOKEN, toggle notebook access).
# Nothing is typed into the notebook body, so a shared .ipynb carries no credential.
try:
    from google.colab import userdata
    HF_TOKEN = (userdata.get("HF_TOKEN") or "").strip()
except Exception:
    HF_TOKEN = ""

if not HF_TOKEN:
    HF_TOKEN = getpass.getpass("Hugging Face token: ").strip()

assert HF_TOKEN, "HF token is required: add HF_TOKEN to Colab Secrets."

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


# ============================================================
# 2. INSTALL KNOWN-WORKING STACK
# ============================================================

print("\n[1/8] Installing vLLM...")

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "vllm==0.29.0",
    ],
    check=True,
)


# Colab's torchaudio may be built against a different CUDA
# version from the PyTorch that vLLM installs. No audio is
# needed here, so remove it.
print("[2/8] Removing incompatible torchaudio...")

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "torchaudio",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)


# Gemma 3 needs torchvision to inspect the architecture, even
# with --language-model-only. Match the PyTorch 2.13 / CUDA 13
# stack that vLLM installs.
print("[3/8] Installing matching torchvision...")

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--force-reinstall",
        "--no-deps",
        "torchvision==0.28.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
    ],
    check=True,
)


# ============================================================
# 3. INSTALL CLOUDFLARED
# ============================================================

print("[4/8] Installing cloudflared...")

subprocess.run(
    "wget -q "
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64.deb "
    "-O /tmp/cloudflared.deb "
    "&& dpkg -i /tmp/cloudflared.deb",
    shell=True,
    check=True,
    stdout=subprocess.DEVNULL,
)


# ============================================================
# 4. VERIFY ENVIRONMENT
# ============================================================

print("[5/8] Verifying environment...\n")

verify = subprocess.run(
    [
        sys.executable,
        "-c",
        """
import torch
import torchvision
import vllm

print("PyTorch     :", torch.__version__)
print("CUDA        :", torch.version.cuda)
print("torchvision :", torchvision.__version__)
print("vLLM        :", vllm.__version__)
print("GPU         :", torch.cuda.get_device_name(0))
print("VRAM (GiB)  :", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
print("BF16        :", torch.cuda.is_bf16_supported())

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
""",
    ],
    capture_output=True,
    text=True,
)

print(verify.stdout)

if verify.returncode != 0:
    print(verify.stderr)
    raise RuntimeError("PyTorch/vLLM environment verification failed.")


# ============================================================
# 5. CLEAN OLD PROCESSES
# ============================================================

subprocess.run(
    [
        "bash",
        "-lc",
        f"fuser -k {PORT}/tcp >/dev/null 2>&1 || true; "
        "pkill -f 'cloudflared tunnel' >/dev/null 2>&1 || true",
    ],
    check=False,
)


# ============================================================
# 6. START vLLM / MEDGEMMA
# ============================================================

LOG_PATH = "/content/vllm-medgemma.log"


# Protects the public endpoint. Generated fresh on every run, so nothing long-lived can
# leak from a shared notebook. It is printed below, and the judge cell picks it up from
# the environment.
API_KEY = secrets.token_urlsafe(32)


cmd = [
    "vllm",
    "serve",
    MODEL,

    "--host",
    "0.0.0.0",

    "--port",
    str(PORT),

    "--served-model-name",
    SERVED_MODEL,

    # ------------------------------------------
    # We only need text.
    # ------------------------------------------
    "--language-model-only",

    # ------------------------------------------
    # Known-working configuration
    # ------------------------------------------
    "--dtype",
    "bfloat16",

    "--max-model-len",
    str(MAX_MODEL_LEN),

    "--max-num-seqs",
    str(MAX_NUM_SEQS),

    "--gpu-memory-utilization",
    "0.94",

    # Protect public endpoint.
    "--api-key",
    API_KEY,
]

if ENFORCE_EAGER:
    cmd.append("--enforce-eager")


print("[6/8] Starting MedGemma...\n")

print("Loading model", end="", flush=True)


log_file = open(LOG_PATH, "w")

server = subprocess.Popen(
    cmd,
    stdout=log_file,
    stderr=subprocess.STDOUT,
    env=os.environ.copy(),
)


# ============================================================
# 7. WAIT FOR MODEL
# ============================================================

health_url = f"http://127.0.0.1:{PORT}/health"

while True:

    # vLLM died
    if server.poll() is not None:

        log_file.flush()

        print("\n\n❌ vLLM exited during startup.\n")

        print("=" * 70)
        print("LAST vLLM LOG LINES")
        print("=" * 70)

        try:
            with open(LOG_PATH, "r") as f:
                lines = f.readlines()

            print("".join(lines[-200:]))

        except Exception as e:
            print("Could not read log:", e)

        raise RuntimeError("vLLM failed to start.")

    # vLLM ready
    try:

        req = urllib.request.Request(
            health_url,
            headers={
                "Authorization": f"Bearer {API_KEY}"
            },
        )

        with urllib.request.urlopen(req, timeout=2) as response:

            if response.status == 200:
                break

    except Exception:
        pass

    print(".", end="", flush=True)
    time.sleep(5)


print("\n✅ MEDGEMMA LOADED")


# ============================================================
# 8. START CLOUDFLARE TUNNEL
# ============================================================

print("\n[7/8] Starting Cloudflare tunnel...")

CF_LOG = "/content/cloudflared.log"
URL_FILE = "/content/public_url.txt"


cf_log_file = open(CF_LOG, "w")

tunnel = subprocess.Popen(
    [
        "cloudflared",
        "tunnel",
        "--no-autoupdate",
        "--url",
        f"http://127.0.0.1:{PORT}",
    ],
    stdout=cf_log_file,
    stderr=subprocess.STDOUT,
)


PUBLIC_URL = None

for _ in range(60):

    time.sleep(2)

    if tunnel.poll() is not None:
        print(open(CF_LOG).read()[-3000:])
        raise RuntimeError("cloudflared exited during startup.")

    match = re.search(
        r"https://[-\w]+\.trycloudflare\.com",
        open(CF_LOG).read(),
    )

    if match:
        PUBLIC_URL = match.group(0)
        break


if not PUBLIC_URL:
    tunnel.kill()
    print(open(CF_LOG).read()[-3000:])
    raise RuntimeError("cloudflared did not produce a URL.")


# The hostname is random on every run, so write it out for
# anything else that needs it.
with open(URL_FILE, "w") as f:
    f.write(PUBLIC_URL)

os.environ["MEDGEMMA_BASE_URL"] = f"{PUBLIC_URL}/v1"
os.environ["MEDGEMMA_API_KEY"] = API_KEY

# What llmjudge.colab.run() reads, so a judge cell in this notebook needs no arguments
# about the server. The URL here is localhost, not the tunnel: a judge in this runtime
# should not leave it and come back through Cloudflare, which cuts any reply the model
# spends more than ~100 s on.
os.environ["LLMJUDGE_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
os.environ["LLMJUDGE_API_KEY"] = API_KEY
os.environ["LLMJUDGE_MODEL"] = SERVED_MODEL


# ============================================================
# DONE
# ============================================================

print("\n[8/8] ✅ READY\n")

print("=" * 72)
print("MEDGEMMA — vLLM — CLOUDFLARE TUNNEL")
print("=" * 72)

print(f"""
Model
  {SERVED_MODEL}

Public URL
  {PUBLIC_URL}

OpenAI-compatible base URL
  {PUBLIC_URL}/v1

Chat endpoint
  {PUBLIC_URL}/v1/chat/completions

Models endpoint
  {PUBLIC_URL}/v1/models

Local base URL (no tunnel, no timeout — prefer this)
  http://127.0.0.1:{PORT}/v1

API key
  {API_KEY}

vLLM log
  {LOG_PATH}

cloudflared log
  {CF_LOG}

URL written to
  {URL_FILE}
""")


# ============================================================
# CURL EXAMPLE
# ============================================================

print("=" * 72)
print("TEST")
print("=" * 72)

print(
f"""
curl '{PUBLIC_URL}/v1/chat/completions' \\
  -H 'Authorization: Bearer {API_KEY}' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "{SERVED_MODEL}",
    "messages": [
      {{
        "role": "user",
        "content": "What are the common causes of hyperkalemia?"
      }}
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }}'
"""
)


# ============================================================
# PYTHON / OPENAI SDK EXAMPLE
# ============================================================

print("=" * 72)
print("OPENAI PYTHON CLIENT — STREAMING")
print("=" * 72)

print(
f"""
# Cloudflare drops any request whose origin stays silent for
# ~100 seconds (error 524). Streaming keeps bytes flowing, so
# long judge responses never trip it.

from openai import OpenAI

client = OpenAI(
    base_url="{PUBLIC_URL}/v1",
    api_key="{API_KEY}",
)

stream = client.chat.completions.create(
    model="{SERVED_MODEL}",
    messages=[
        {{
            "role": "user",
            "content": "What are the common causes of hyperkalemia?"
        }}
    ],
    temperature=0.2,
    max_tokens=512,
    stream=True,
)

text = "".join(
    chunk.choices[0].delta.content or ""
    for chunk in stream
)

print(text)
"""
)
