#!/usr/bin/env python3
"""
Brain Tumor Classification CNN Model Training (Minimal Version)
Testing basic setup before adding ROI preprocessing
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import json
from datetime import datetime

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
DATASET_DIR = Path('dataset/brain')
TRAIN_DIR = DATASET_DIR / 'Training'
TEST_DIR = DATASET_DIR / 'Testing'

def create_basic_datagen():
    """Create basic data generators without complex preprocessing."""
    print("Creating basic data generators...")

    # Basic data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("✓ Basic data generators created")
    return train_generator, validation_generator, test_generator

def build_basic_model():
    """Build basic EfficientNetB0 model."""
    print("Building basic EfficientNetB0 model...")

    # Load EfficientNetB0 base
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False

    # Build model
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✓ Basic model built")
    return model

def train_basic_model():
    """Train the basic model."""
    print("="*80)
    print("BRAIN TUMOR CLASSIFICATION CNN MODEL TRAINING (BASIC VERSION)")
    print("="*80)

    # Check dataset
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        print("❌ Dataset not found!")
        return

    print("✓ Dataset found")

    # Create data generators
    train_gen, val_gen, test_gen = create_basic_datagen()

    # Build model
    model = build_basic_model()

    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/brain_tumor_basic.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Train
    print("Starting basic training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # Evaluate
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test accuracy: {test_accuracy:.4f}")
    # Save metadata
    metadata = {
        'model_type': 'basic_efficientnet',
        'final_accuracy': float(test_accuracy),
        'training_history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        },
        'training_config': {
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': len(history.history['accuracy'])
        }
    }

    with open('models/brain_tumor_basic_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✓ Basic training completed!")
    return model, history, metadata

if __name__ == "__main__":
    train_basic_model()