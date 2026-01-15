import os
import random
import cv2
import matplotlib.pyplot as plt

def test_vision():
    data_dir = '../data'
    categories = ['Trees', 'NoTrees']
    
    print("--- GreenWatch Vision Test ---")
    
    category = random.choice(categories)
    folder_path = os.path.join(data_dir, category)
    
    try:
        images = os.listdir(folder_path)
        if not images:
            print(f"Error: No images found in {folder_path}")
            return
            
        random_image = random.choice(images)
        image_path = os.path.join(folder_path, random_image)
        
        img_array = cv2.imread(image_path)
        
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        print(f"✅ Success! Loading a sample from: {category}")
        print(f"   Image Name: {random_image}")
        print(f"   Image Size: {img_array.shape} (Height, Width, Color Channels)")
        
        plt.imshow(img_rgb)
        plt.title(f"Satellite View: {category}")
        plt.axis('off')
        plt.show()

    except FileNotFoundError:
        print("❌ Error: Could not find the data folder.")
        print("Make sure your folder structure is: GreenWatch/data/Trees/...")

if __name__ == "__main__":
    test_vision()