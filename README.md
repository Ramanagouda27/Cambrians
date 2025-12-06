# Acuity Drive – Smart Vehicle Mirror (Driver Co-Pilot System)

This repository contains AI-powered night driving assistance using YOLOv8, OpenCV, glare removal, lane detection, and distance estimation.

## Features
- Glare removal
- Lane detection
- Vehicle detection (YOLOv8)
- Distance estimation
- Alerts system

## Project Structure
See folders: src/, models/, data/, scripts/

## Model weights

This repo does **not** include the heavy model weights. Download the YOLO weights and place them inside the `models/` folder.

- YOLOv8n: https://drive.google.com/file/d/12asbKzYYOjbIO8lNRpvIiflUJzL2-4Dm/view?usp=sharing  
  Direct download link: `https://drive.google.com/uc?export=download&id=12asbKzYYOjbIO8lNRpvIiflUJzL2-4Dm`

After downloading, create a folder:

models/
└── yolov8n.pt


## How to Run
1. python -m venv venv
2. Activate the venv
3. pip install -r requirements.txt
4. python src/run.py --demo glare --camera 0
