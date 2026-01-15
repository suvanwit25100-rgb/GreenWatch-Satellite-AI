import tensorflow as tf
from tensorflow.keras import layers, models
import os

def train_brain():
    print("--- 🧠 GreenWatch: Initializing Training ---")

    DATA_DIR = '../data'
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    BATCH_SIZE = 32

    print("Loading images...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("Building Neural Network with Augmentation...")
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_augmentation,
        
        layers.Rescaling(1./255),
        
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\n--- Starting Training (This may take 2-5 mins) ---")
    epochs = 10 
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save('../models/greenwatch_model.h5')
    print("\n✅ Success! Model saved to ../models/greenwatch_model.h5")

if __name__ == "__main__":
    if not os.path.exists('../models'):
        os.makedirs('../models')
    train_brain()