import tensorflow as tf
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

def predict_random_image():
    model_path = '../models/greenwatch_model.h5'
    if not os.path.exists(model_path):
        print("❌ Model not found! Wait for training to finish first.")
        return

    print("Loading AI Brain...")
    model = tf.keras.models.load_model(model_path)

    base_dir = '../data'
    category = random.choice(['Trees', 'NoTrees'])
    folder = os.path.join(base_dir, category)
    
    if not os.listdir(folder):
        print("Error: Folder is empty.")
        return
        
    filename = random.choice(os.listdir(folder))
    img_path = os.path.join(folder, filename)

    print(f"Testing on image: {filename} (Actual: {category})")

    img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0][0]

    if score > 0.5:
        result = "FOREST DETECTED 🌲"
        confidence = score * 100
    else:
        result = "DEFORESTATION / BARREN ⚠️"
        confidence = (1 - score) * 100

    print("\n" + "="*30)
    print(f"AI ANALYSIS: {result}")
    print(f"Confidence:  {confidence:.2f}%")
    print("="*30)

    display_img = cv2.imread(img_path)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(display_img)
    plt.title(f"AI Says: {result}\n({confidence:.1f}% Confidence)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_random_image()