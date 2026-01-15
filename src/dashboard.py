import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import random

st.set_page_config(page_title="GreenWatch AI", page_icon="🛰️")

st.title("🛰️ GreenWatch: Deforestation Detector")
st.markdown("Use Artificial Intelligence to detect **Illegal Logging** and **Deforestation** in satellite imagery.")

@st.cache_resource
def load_model():
    model_path = '../models/greenwatch_model.h5'
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

if model is None:
    st.error("❌ Model not found! Please run 'train_cnn.py' first.")
    st.stop()

st.sidebar.header("Control Panel")
option = st.sidebar.radio("Choose Input Method:", ["Random Test Image", "Upload Image"])

def make_prediction(img_array, original_image):
    img_array = tf.expand_dims(img_array, 0) 
    
    prediction = model.predict(img_array)
    score = prediction[0][0]

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption="Satellite View", use_container_width=True)

    with col2:
        st.subheader("AI Analysis")
        if score > 0.5:
            confidence = score * 100
            st.success(f"🌲 **FOREST DETECTED**")
            st.metric("Confidence", f"{confidence:.2f}%")
        else:
            confidence = (1 - score) * 100
            st.error(f"⚠️ **DEFORESTATION / BARREN**")
            st.metric("Confidence", f"{confidence:.2f}%")
            st.write("Action: Flagging coordinates for review.")


if option == "Random Test Image":
    if st.button("🎲 Analyze Random Satellite Image"):
        base_dir = '../data'
        category = random.choice(['Trees', 'NoTrees'])
        folder = os.path.join(base_dir, category)
        
        if os.path.exists(folder):
            filename = random.choice(os.listdir(folder))
            img_path = os.path.join(folder, filename)
            
            image = Image.open(img_path)
            image = image.resize((64, 64))
            img_array = np.array(image)
            
            st.info(f"Loaded sample from category: **{category}**")
            make_prediction(img_array, image)
        else:
            st.error("Data folder not found. Check your path.")

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a satellite photo...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.write("Processing image...")
        
        img_resized = image.resize((64, 64))
        img_array = np.array(img_resized)
        
        make_prediction(img_array, image)