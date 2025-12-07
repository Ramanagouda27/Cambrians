# scripts/get_model.py
import os, urllib.parse, sys, subprocess, pathlib

MODEL_URL = "https://drive.google.com/uc?export=download&id=12asbKzYYOjbIO8lNRpvIiflUJzL2-4Dm"
OUT = pathlib.Path("models")
OUT.mkdir(exist_ok=True)
out_file = OUT / "yolov8n.pt"

if out_file.exists():
    print("Model already exists:", out_file)
    sys.exit(0)

print("Downloading model to", out_file)
# Use curl / wget fallback for convenience:
try:
    subprocess.check_call(["curl", "-L", MODEL_URL, "-o", str(out_file)])
except Exception:
    try:
        subprocess.check_call(["wget", MODEL_URL, "-O", str(out_file)])
    except Exception:
        print("Please download manually from", MODEL_URL)
