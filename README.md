# 🔥 Acuity Drive – Smart Vehicle Mirror (Driver Co-Pilot System)

AI-powered night-driving assistance using **YOLOv8**, **OpenCV**, and custom perception modules for:

- 🌟 Glare removal  
- 🛣️ Lane detection  
- 🚗 Vehicle detection  
- 📏 Distance estimation  
- 💥 Rear collision alerts  
- 🪞 Side-mirror blind-spot alerts  

This system works on **video files or live webcam** and is designed to improve safety on Indian nighttime roads.

---

## 📂 Project Structure

```
Acuity-Drive/
│
├── src/                     # All main Python scripts
│   ├── run.py
│   ├── glare_demo.py
│   ├── multi_lane_detect.py
│   ├── rear_ttc_demo.py
│   └── side_mirror_video_alert.py
│
├── data/                    # Your test videos (add manually)
│
├── models/                  # YOLO model weights (downloaded separately)
│
├── scripts/
│   └── get_model.py         # Auto-download YOLOv8n model
│
├── requirements.txt
└── README.md
```

---

## ⚠️ Model Weights (Not Included in Repo)

The YOLOv8 model file is large and cannot be committed to GitHub.

### 🔗 Download YOLOv8n Model (Google Drive)
https://drive.google.com/drive/folders/15KpNH0DAUecHDORTcbkdwvGRoaC581pK?usp=sharing

After downloading, put the file here:

```
models/
 └── yolov8n.pt
```

---

## 🟦 Automatic Model Download (Recommended)

Instead of manually downloading, simply run:

```bash
python scripts/get_model.py
```

This creates:

```
models/yolov8n.pt
```

If auto-download fails, manually place the model file.

---

# 🚀 How to Run This Project (Judges / Teammates / New System)

These steps work on **any machine**.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ramanagouda27/Cambrians.git
cd Cambrians
```

---

## 2️⃣ Checkout the Development Branch

```bash
git fetch origin
git checkout dev
```

---

## 3️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Prompt will change to:

```
(venv)
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5️⃣ Download the YOLO Model

```bash
python scripts/get_model.py
```

Verify:

```bash
dir models
```

You should see:

```
yolov8n.pt
```

---

# ▶️ Running the System

---

## 🔦 1. Run Glare Removal

### 🟢 Using Webcam

```bash
python src/run.py --demo glare --camera 0
```

---

### 🎥 Using Video File

Place your video inside:

```
data/
```

Example:

```
data/3_lane_kelly_dashcam.mp4
```

Then run:

```bash
python src/glare_demo.py --video "data/3_lane_kelly_dashcam.mp4"
```

---

## 🛣️ 2. Multi-Lane Detection

```bash
python src/multi_lane_detect.py --video "data/3_lane_kelly_dashcam.mp4"
```

---

## 🚗 3. Rear TTC Collision Warning

```bash
python src/rear_ttc_demo.py --video "data/3_lane_kelly_dashcam.mp4"
```

---

## 🪞 4. Side-Mirror Blind-Spot Alert

```bash
python src/side_mirror_video_alert.py --video "data/3_lane_kelly_dashcam.mp4"
```

---

## 🎥 Sample Videos (Download to Test the System)

This project requires video input. Download sample test videos from the link below:

🔗 **Sample Dashcam Videos (Google Drive)**  
https://drive.google.com/drive/folders/15KpNH0DAUecHDORTcbkdwvGRoaC581pK?usp=sharing

After downloading, place the videos inside:



# 📝 Notes for Judges

- Requires **Python 3.10+**
- YOLO model downloads **automatically** via script
- Video files are **not included** — add any MP4 to the `data/` folder
- All modules run offline using CPU (GPU optional
