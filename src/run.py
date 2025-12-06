import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_MAP = {
    "glare": "glare_demo.py",
    "lane":  "multi_lane_detect.py",
    "rear":  "rear_ttc_demo.py",
    "side":  "side_mirror_video_alert.py"
}

def main():
    parser = argparse.ArgumentParser(description="Acuity Drive - Demo Launcher (safe)")
    parser.add_argument("--demo", choices=DEMO_MAP.keys(), default="glare", help="Which demo to run")
    parser.add_argument("--model", help="Path to model file (optional)", default=str(ROOT.parent / "models" / "yolov8n.pt"))
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    script = ROOT / DEMO_MAP[args.demo]
    if not script.exists():
        print(f"Demo script not found: {script}")
        return

    # Build command to run the demo as a new Python process
    cmd = [sys.executable, str(script)]
    # Pass model & camera as optional CLI args so demo scripts can accept them if implemented
    cmd += ["--model", args.model, "--camera", str(args.camera)]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
