🔥 Acuity Drive – Smart Vehicle Mirror (Driver Co-Pilot System)

AI-powered night-driving assistance using YOLOv8, OpenCV, and custom perception modules for:

Glare removal

Lane detection

Vehicle detection

Distance estimation

Rear collision alerts

Side-mirror blind-spot alerts

This system works on video files or live camera and is built to improve safety on Indian nighttime roads.

📂 Project Structure
Acuity-Drive/
│
├── src/                # All main Python scripts
│   ├── run.py
│   ├── glare_demo.py
│   ├── multi_lane_detect.py
│   ├── rear_ttc_demo.py
│   └── side_mirror_video_alert.py
│
├── data/               # Your test videos (add manually)
│
├── models/             # YOLO model weights (downloaded separately)
│
├── scripts/
│   └── get_model.py    # Auto-download YOLOv8n model
│
├── requirements.txt
└── README.md

⚠️ Model Weights (Not Included in Repo)

The YOLOv8 model file is large and cannot be committed to GitHub.

Download it here:

🔗 YOLOv8n Model (Google Drive)

https://drive.google.com/drive/folders/15KpNH0DAUecHDORTcbkdwvGRoaC581pK?usp=sharing

After downloading, place it inside the models/ folder:

models/
 └── yolov8n.pt

🟦 Automatic Model Download (Recommended)

Instead of manual download, run:

python scripts/get_model.py


This creates:

models/yolov8n.pt


If the script fails, download from the Drive link above and place manually.

🚀 How to Run This Project (Clean Setup Instructions)

These steps will work on any machine (judges, teammates, or fresh system).

1️⃣ Clone the Repository
git clone https://github.com/Ramanagouda27/Cambrians.git
cd Cambrians

2️⃣ Checkout the Development Branch
git fetch origin
git checkout dev

3️⃣ Create and Activate Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1


If PowerShell blocks activation, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1


You should now see:

(venv)

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Download the YOLO Model
python scripts/get_model.py


Verify:

dir models


You should see:

yolov8n.pt

▶️ Running the System
🔦 1. Run Glare Removal
Using webcam
python src\run.py --demo glare --camera 0

🎥 Using a video file

👉 Place your sample video inside the data/ folder:

Example:

data/3_lane_kelly_dashcam.mp4


Then run:

python src\glare_demo.py --video "data/3_lane_kelly_dashcam.mp4"

🛣️ 2. Run Multi-Lane Detection
python src\multi_lane_detect.py --video "data/3_lane_kelly_dashcam.mp4"

🚗 3. Rear TTC Collision Warning
python src\rear_ttc_demo.py --video "data/3_lane_kelly_dashcam.mp4"

🪞 4. Side-Mirror Blind-Spot Alert
python src\side_mirror_video_alert.py --video "data/3_lane_kelly_dashcam.mp4"

📝 Notes for Judges

This project requires Python 3.10+.

YOLO model downloads automatically via script.

Video files are not included — please add any MP4 into data/ folder.

All detections run offline on CPU (GPU optional).

🏆 Authors

Team Cambrians
Developed for NxtWave Buildathon.
