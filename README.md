# 🖐️ ARKit LiDAR-Based 3D Hand Tracking Pipeline

## 📌 Overview
This project implements a 3D hand tracking pipeline using ARKit and LiDAR depth data. It reconstructs hand poses in camera and world space and generates visualization outputs.

## 🧠 Features
- 2D → 3D reprojection using depth maps
- Wrist-to-head distance estimation
- Camera pose interpolation (SLERP)
- Depth overlay visualization
- 3D world trajectory visualization
- Reprojection error analysis

## 🏗️ Pipeline Workflow
ARKit Input → Depth Extraction → Pose Interpolation → 3D Reconstruction → Distance Estimation → Visualization

## 📂 Structure
project/
├── data/
├── src/
│   └── pipeline.py
├── assets/
├── README.md
├── requirements.txt
└── .gitignore

## ⚙️ Installation
pip install -r requirements.txt

## ▶️ Run
python src/pipeline.py --raw-dir "data/sample_input" --output-dir "data/sample_output"

## 📊 Outputs
- Depth overlay video
- 3D visualization video
- Unified JSON dataset

## 🎥 Pipeline Outputs

<p align="center">
  <b>Depth Overlay</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>3D Visualization</b>
</p>

<p align="center">
  <img src="assets/video1.gif" width="48%" />
  <img src="assets/video2.gif" width="48%" />
</p>

<p align="center">
  <b>Combined Output</b>
</p>

<p align="center">
  <img src="assets/video3.gif" width="80%" />
</p>

## ⚠️ Note
Large files (videos, depth maps) are excluded.

## 👨‍💻 Author
Pavan Nanjunda
