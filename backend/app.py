"""
GreenWatch Backend — Flask API
Serves the TensorFlow CNN model for satellite image classification.
Falls back to demo mode when the model file is unavailable.
"""

import os
import sys
import random
import base64
import io
import json
import time
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Paths (relative to backend/)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'greenwatch_model.h5')
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')

IMG_SIZE = (64, 64)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
model = None
demo_mode = True

def load_model():
    global model, demo_mode
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            demo_mode = False
            print("✅ Model loaded successfully")
        else:
            print(f"⚠️  Model file not found at {MODEL_PATH} — running in DEMO mode")
    except Exception as e:
        print(f"⚠️  Could not load model: {e} — running in DEMO mode")

load_model()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def predict_image(img: Image.Image):
    """Run inference on a PIL image. Returns (label, confidence, raw_score)."""
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)

    if model is not None:
        prediction = model.predict(img_array, verbose=0)
        raw_score = float(prediction[0][0])
    else:
        # Demo mode — synthetic prediction
        raw_score = random.uniform(0.05, 0.95)

    if raw_score > 0.5:
        label = "Forest"
        confidence = raw_score * 100
    else:
        label = "Deforested"
        confidence = (1 - raw_score) * 100

    return label, round(confidence, 2), round(raw_score, 4)


def image_to_base64(img: Image.Image, max_size=400):
    """Convert PIL Image to base64 string for JSON transport."""
    img_copy = img.copy()
    img_copy.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    img_copy.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "demo_mode": demo_mode,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "architecture": "Sequential CNN",
            "input_shape": "64 × 64 × 3",
            "layers": [
                {"name": "Data Augmentation", "type": "augmentation", "detail": "RandomFlip + RandomRotation(0.2)"},
                {"name": "Rescaling", "type": "preprocessing", "detail": "1/255 normalization"},
                {"name": "Conv2D-16", "type": "conv", "detail": "16 filters, 3×3, ReLU, same padding"},
                {"name": "MaxPooling2D", "type": "pool", "detail": "2×2 pool"},
                {"name": "Conv2D-32", "type": "conv", "detail": "32 filters, 3×3, ReLU, same padding"},
                {"name": "MaxPooling2D", "type": "pool", "detail": "2×2 pool"},
                {"name": "Conv2D-64", "type": "conv", "detail": "64 filters, 3×3, ReLU, same padding"},
                {"name": "MaxPooling2D", "type": "pool", "detail": "2×2 pool"},
                {"name": "Flatten", "type": "reshape", "detail": "→ 1D vector"},
                {"name": "Dense-128", "type": "dense", "detail": "128 units, ReLU"},
                {"name": "Dense-1", "type": "output", "detail": "1 unit, Sigmoid (binary)"},
            ],
            "optimizer": "Adam",
            "loss": "Binary Crossentropy",
            "training_epochs": 10,
        }
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    start = time.time()
    label, confidence, raw_score = predict_image(img)
    inference_ms = round((time.time() - start) * 1000, 1)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "raw_score": raw_score,
        "inference_ms": inference_ms,
        "demo": demo_mode,
        "image_b64": image_to_base64(img),
        "timestamp": datetime.now().isoformat(),
    })


@app.route('/api/random-sample', methods=['GET'])
def random_sample():
    categories = ['Trees', 'NoTrees']
    
    # Check if data dir exists
    if not os.path.exists(DATA_DIR):
        # Generate a synthetic response in demo mode
        label = random.choice(["Forest", "Deforested"])
        confidence = round(random.uniform(72, 98), 2)
        raw_score = confidence / 100 if label == "Forest" else 1 - (confidence / 100)
        return jsonify({
            "label": label,
            "confidence": confidence,
            "raw_score": round(raw_score, 4),
            "actual_category": random.choice(categories),
            "filename": "demo_sample.jpg",
            "inference_ms": round(random.uniform(15, 60), 1),
            "demo": True,
            "image_b64": None,
            "timestamp": datetime.now().isoformat(),
        })

    category = random.choice(categories)
    folder = os.path.join(DATA_DIR, category)

    if not os.path.exists(folder) or not os.listdir(folder):
        return jsonify({"error": f"No images in {category} folder"}), 404

    filename = random.choice(os.listdir(folder))
    img_path = os.path.join(folder, filename)

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return jsonify({"error": "Could not open image"}), 500

    start = time.time()
    label, confidence, raw_score = predict_image(img)
    inference_ms = round((time.time() - start) * 1000, 1)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "raw_score": raw_score,
        "actual_category": category,
        "filename": filename,
        "inference_ms": inference_ms,
        "demo": demo_mode,
        "image_b64": image_to_base64(img),
        "timestamp": datetime.now().isoformat(),
    })


@app.route('/api/predict-location', methods=['GET'])
def predict_location():
    try:
        lat = float(request.args.get('lat', 0))
        lng = float(request.args.get('lng', 0))
    except ValueError:
        return jsonify({"error": "Invalid coordinates"}), 400

    start = time.time()
    
    # 0.005 degrees is roughly 500m across
    delta = 0.005
    bbox = f"{lng-delta},{lat-delta},{lng+delta},{lat+delta}"
    url = f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export?bbox={bbox}&bboxSR=4326&imageSR=4326&size=64,64&f=image"
    
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        img_data = response.read()
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Failed to fetch satellite imagery: {str(e)}"}), 502

    label, confidence, raw_score = predict_image(img)
    inference_ms = round((time.time() - start) * 1000, 1)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "raw_score": raw_score,
        "inference_ms": inference_ms,
        "demo": demo_mode,
        "image_b64": image_to_base64(img),
        "timestamp": datetime.now().isoformat(),
        "lat": lat,
        "lng": lng
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Return dashboard statistics (simulated for demo)."""
    return jsonify({
        "total_scans": random.randint(1240, 1580),
        "forest_detected": random.randint(820, 1060),
        "deforestation_alerts": random.randint(180, 320),
        "avg_confidence": round(random.uniform(88.0, 96.5), 1),
        "coverage_km2": round(random.uniform(12400, 18700), 0),
        "model_accuracy": 93.5,
        "regions_monitored": 24,
        "active_satellites": 3,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n🛰️  GreenWatch API starting...")
    print(f"   Model: {'✅ Loaded' if not demo_mode else '⚠️  Demo Mode'}")
    print(f"   Frontend: {FRONTEND_DIR}")
    print(f"   Server: http://localhost:5001\n")
    app.run(debug=True, port=5001)
