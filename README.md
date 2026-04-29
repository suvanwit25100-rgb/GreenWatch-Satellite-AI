# 🛰️ GreenWatch: AI Deforestation Detector

**GreenWatch** is a modern, real-time application that uses computer vision to detect illegal logging and deforestation in satellite imagery. 

Powered by a custom **Trained Brain** (via Roboflow Inference), this tool analyzes satellite photos in real-time to classify land as either **"Forest"** or **"Deforested/Barren"**. It features a premium full-stack web dashboard with interactive Leaflet map scanning, animated confidence gauges, and drag-and-drop image analysis.

## 🚀 Features
* **Trained Brain Model:** A custom-trained computer vision model hosted on Roboflow, optimized for satellite texture recognition.
* **Interactive Map Scanning:** Click anywhere on the map to fetch real-time satellite imagery via the Esri API and scan it for deforestation instantly.
* **Premium Web Dashboard:** Dark satellite command center UI with glassmorphism design, real-time analysis, and animated visualizations.
* **Drag & Drop Upload:** Upload satellite imagery directly from your desktop for instant classification.
* **Animated Confidence Gauge:** Canvas-rendered arc gauge with smooth animation showing prediction certainty.
* **Prediction History:** Session log of all analysis results with thumbnails and metadata.
* **Demo Mode:** Fallback synthetic predictions so the UI can still be showcased if API connectivity fails.
* **RESTful API:** Flask backend exposes clean JSON endpoints for model inference.

## 🛠️ Tech Stack
* **AI / Computer Vision:** Roboflow Inference SDK (`inference-sdk`)
* **Image Processing:** Pillow, NumPy
* **Backend API:** Flask, Flask-CORS, Gunicorn
* **Frontend:** Vanilla HTML5, CSS3, JavaScript (Canvas API)
* **Maps:** Leaflet.js, Esri World Imagery
* **Deployment:** Ready for Render (includes `render.yaml`, `Procfile`, and `.python-version`)

## 📂 Project Structure
```text
GreenWatch-Satellite-AI/
├── backend/
│   └── app.py                 # Flask API serving the Roboflow inference client
├── frontend/
│   ├── index.html             # Premium dashboard UI
│   ├── style.css              # Satellite command center theme
│   └── app.js                 # Client logic, map initialization & canvas visualizations
├── requirements.txt           # Lean Python dependencies
├── Procfile                   # Deployment start command
├── render.yaml                # Render Blueprint configuration
├── runtime.txt                # Python version for Render
├── .python-version            # Python version pin (3.11.0)
└── README.md
```

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard
```bash
python backend/app.py
```
Open **http://localhost:5001** in your browser.

## 🌍 Live Deployment (Render)
This repository is pre-configured for 1-click deployment on Render.
1. Create a New Web Service on Render and connect this repository.
2. Render will automatically detect the settings from `render.yaml`.
3. The app is served via `gunicorn` on port `$PORT`.

## 🖥️ API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System status + trained brain workflow info |
| `/api/predict` | POST | Upload image for classification |
| `/api/random-sample` | GET | Classify a random demo image |
| `/api/predict-location` | GET | Fetch and classify satellite imagery for given `lat` and `lng` |
| `/api/stats` | GET | Dashboard analytics data |

## 📸 Screenshots

_Launch the dashboard and explore the satellite command center interface!_
