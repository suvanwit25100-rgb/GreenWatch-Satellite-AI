# 🛰️ GreenWatch: AI Deforestation Detector

**GreenWatch** is a Deep Learning application that uses computer vision to detect illegal logging and deforestation in satellite imagery. 

Built with **TensorFlow (CNN)**, this tool analyzes satellite photos in real-time to classify land as either **"Forest"** or **"Deforested/Barren"**, providing a scalable way to monitor environmental changes. Features a premium fullstack web dashboard with interactive CNN architecture visualization, animated confidence gauges, and drag-and-drop image analysis.

## 🚀 Features
* **Convolutional Neural Network (CNN):** A custom-trained deep learning model optimized for satellite texture recognition.
* **Data Augmentation:** Implements random flips and rotations during training to make the model robust against different viewing angles.
* **Premium Web Dashboard:** Dark satellite command center UI with glassmorphism design, real-time analysis, and animated visualizations.
* **Drag & Drop Upload:** Upload satellite imagery directly from your desktop for instant classification.
* **Animated Confidence Gauge:** Canvas-rendered arc gauge with smooth animation showing prediction certainty.
* **CNN Architecture Visualizer:** Interactive layer-by-layer view of the model's structure with color-coded layer types.
* **Prediction History:** Session log of all analysis results with thumbnails and metadata.
* **Demo Mode:** Full UI experience even without a trained model — perfect for showcasing.
* **RESTful API:** Flask backend exposes clean JSON endpoints for model inference.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Image Processing:** OpenCV, PIL
* **Backend API:** Flask, Flask-CORS
* **Frontend:** Vanilla HTML5, CSS3, JavaScript (Canvas API)
* **Visualization:** Animated gauges, architecture diagrams, counter animations
* **Language:** Python 3.8+

## 📂 Project Structure
```text
GreenWatch-Satellite-AI/
├── data/                      # Satellite image dataset (Trees vs NoTrees)
├── models/                    # Saved .h5 trained models
├── src/
│   ├── train_cnn.py           # Script to build and train the CNN
│   ├── predict_forest.py      # CLI script to test single images
│   ├── check_setup.py         # Verify data pipeline setup
│   └── dashboard.py           # Legacy Streamlit interface
├── backend/
│   └── app.py                 # Flask API serving the TF model
├── frontend/
│   ├── index.html             # Premium dashboard UI
│   ├── style.css              # Satellite command center theme
│   └── app.js                 # Client logic & canvas visualizations
├── requirements.txt           # Python dependencies
└── README.md
```

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (Optional)
Place satellite images under:
```
data/Trees/       # Forest images
data/NoTrees/     # Deforested / barren images
```

### 3. Train the Model (Optional)
```bash
cd src
python train_cnn.py
```
Training takes ~2-5 mins and saves the model to `models/greenwatch_model.h5`.

### 4. Launch the Dashboard
```bash
python backend/app.py
```
Open **http://localhost:5000** in your browser.

> **Note:** The dashboard works in **Demo Mode** even without training data or a saved model — it simulates predictions so you can explore the full interface.

## 🖥️ API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System status + model architecture info |
| `/api/predict` | POST | Upload image for classification |
| `/api/random-sample` | GET | Classify a random image from the dataset |
| `/api/stats` | GET | Dashboard analytics data |

## 📸 Screenshots

_Launch the dashboard and explore the satellite command center interface!_
