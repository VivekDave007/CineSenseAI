import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/archive_5/images/Images')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'dog_classifier.keras')
CLASSES_SAVE_PATH = os.path.join(MODEL_DIR, 'dog_classes.json')

IMG_SIZE = (128, 128) # Downsized to strictly prevent CPU OOM
BATCH_SIZE = 4 # Downsized batch to prevent CPU OOM
EPOCHS = 1 # Force 1 epoch for quick compile compilation

def train_model():
    print(f"Loading Stanford Dogs dataset from: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory {DATA_DIR} does not exist.")
        return

    # 1. Data Augmentation & Preprocessing
    # MobileNetV2 expects pixels in [-1, 1], so we use its specific preprocess_input
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    # Validation split so we can measure accuracy
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        fill_mode='nearest'
    )

    print("Preparing Training Data Generator...")
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    print("Preparing Validation Data Generator...")
    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Save the class mapping (e.g. integer index -> breed name)
    # The folder names are like "n02085620-Chihuahua"
    class_indices = train_generator.class_indices
    # Reverse to get index: class_name
    idx_to_class = {v: k.split('-')[-1].replace('_', ' ').title() for k, v in class_indices.items()}
    
    with open(CLASSES_SAVE_PATH, 'w') as f:
        json.dump(idx_to_class, f)
    print(f"Saved {len(idx_to_class)} class mappings to {CLASSES_SAVE_PATH}")

    # 2. Build MobileNetV2 Model
    print("Building MobileNetV2 Transfer Learning Architecture...")
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Freeze the base model (we only train the top layer first)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x) # Prevent overfitting
    predictions = Dense(len(idx_to_class), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stop = EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy')
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-5)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')

    # 3. Train
    print("Starting Training (Fast-tracked for UI prototype compile)...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        steps_per_epoch=2, # Tiny step to instantly finish epoch 1
        validation_steps=1,
        callbacks=[checkpoint]
    )

    print(f"\nTraining Complete! Best model saved to {MODEL_SAVE_PATH}")
    
    # Quick sanity check eval
    val_loss, val_acc = model.evaluate(validation_generator)
    print(f"Final Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    train_model()
