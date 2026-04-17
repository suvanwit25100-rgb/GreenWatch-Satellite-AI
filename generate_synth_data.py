import os
import random
from PIL import Image, ImageDraw

def generate_synthetic_data(base_dir='data', samples_per_class=100, img_size=(64, 64)):
    """Generates synthetic "Trees" and "NoTrees" images so the ML pipeline can be run."""
    categories = ['Trees', 'NoTrees']
    for cat in categories:
        os.makedirs(os.path.join(base_dir, cat), exist_ok=True)
    
    # Generate Trees (mostly green variations)
    print("Generating synthetic 'Trees' dataset...")
    for i in range(samples_per_class):
        img = Image.new('RGB', img_size)
        draw = ImageDraw.Draw(img)
        # Base green
        base_green = (random.randint(10, 40), random.randint(80, 160), random.randint(10, 50))
        img.paste(base_green, [0, 0, img_size[0], img_size[1]])
        
        # Draw some noisy circles/polygons to simulate treetops
        for _ in range(random.randint(20, 50)):
            x = random.randint(0, img_size[0])
            y = random.randint(0, img_size[1])
            r = random.randint(2, 10)
            col = (random.randint(0, 50), random.randint(60, 200), random.randint(0, 50))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=col)
        
        img.save(os.path.join(base_dir, 'Trees', f'synthetic_tree_{i:03d}.jpg'))

    # Generate NoTrees (mostly brown/tan/grey variations)
    print("Generating synthetic 'NoTrees' dataset...")
    for i in range(samples_per_class):
        img = Image.new('RGB', img_size)
        draw = ImageDraw.Draw(img)
        # Base brown/grey
        r_val = random.randint(150, 220)
        g_val = random.randint(120, 180)
        b_val = random.randint(80, 140)
        img.paste((r_val, g_val, b_val), [0, 0, img_size[0], img_size[1]])
        
        # Add some rocky speckles/noise
        for _ in range(random.randint(30, 80)):
            x = random.randint(0, img_size[0])
            y = random.randint(0, img_size[1])
            col = (random.randint(100, 255), random.randint(100, 200), random.randint(80, 150))
            draw.point((x, y), fill=col)
            if random.random() > 0.5:
                draw.line([(x, y), (x+random.randint(1,5), y+random.randint(-2,2))], fill=col)
                
        img.save(os.path.join(base_dir, 'NoTrees', f'synthetic_barren_{i:03d}.jpg'))
    
    print("✅ Synthetic dataset created successfully in data/")

if __name__ == "__main__":
    generate_synthetic_data()
