# 🛰️ GreenWatch: AI Deforestation Detector

**GreenWatch** is a Deep Learning application that uses computer vision to detect illegal logging and deforestation in satellite imagery. 

Built with **TensorFlow (CNN)** and **Streamlit**, this tool analyzes satellite photos in real-time to classify land as either "Forest" or "Deforested/Barren," providing a scalable way to monitor environmental changes.

## 🚀 Features
* **Convolutional Neural Network (CNN):** A custom-trained deep learning model optimized for satellite texture recognition.
* **Data Augmentation:** Implements random flips and rotations during training to make the model robust against different viewing angles.
* **Interactive Dashboard:** A user-friendly web interface built with **Streamlit** to upload images or test random samples.
* **Real-time Confidence Score:** Provides a probability percentage (0-100%) for every prediction.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Image Processing:** OpenCV, PIL
* **Visualization:** Streamlit, Matplotlib
* **Language:** Python 3.8+

## 📂 Project Structure
```text
GreenWatch/
├── data/                  # Satellite image dataset (Trees vs NoTrees)
├── models/                # Saved .h5 trained models
├── src/
│   ├── train_cnn.py       # Script to build and train the CNN
│   ├── predict_forest.py  # Script to test single images
│   └── dashboard.py       # The web application interface
└── requirements.txt       # List of dependencies
