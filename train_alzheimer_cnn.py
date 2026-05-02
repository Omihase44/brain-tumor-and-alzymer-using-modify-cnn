#!/usr/bin/env python3
"""
Alzheimer Detection CNN Training Script
Adapted from alzheimer-detection-using-cnn.ipynb for local dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
import os
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(data_dir):
    """Load images from local dataset directory"""
    non_demented = []
    very_mild_demented = []
    mild_demented = []
    moderate_demented = []

    # Load images from each category
    for dirname, _, filenames in os.walk(os.path.join(data_dir, 'Non Demented')):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                non_demented.append(os.path.join(dirname, filename))

    for dirname, _, filenames in os.walk(os.path.join(data_dir, 'Very mild Dementia')):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                very_mild_demented.append(os.path.join(dirname, filename))

    for dirname, _, filenames in os.walk(os.path.join(data_dir, 'Mild Dementia')):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                mild_demented.append(os.path.join(dirname, filename))

    for dirname, _, filenames in os.walk(os.path.join(data_dir, 'Moderate Dementia')):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                moderate_demented.append(os.path.join(dirname, filename))

    logger.info(f"Loaded {len(non_demented)} Non Demented images")
    logger.info(f"Loaded {len(mild_demented)} Mild Dementia images")
    logger.info(f"Loaded {len(moderate_demented)} Moderate Dementia images")
    logger.info(f"Loaded {len(very_mild_demented)} Very mild Dementia images")

    return non_demented, mild_demented, moderate_demented, very_mild_demented

def preprocess_images(image_lists, target_size=(128, 128)):
    """Preprocess images and create dataset"""
    encoder = OneHotEncoder()
    encoder.fit([[0], [1], [2], [3]])  # 0: non_demented, 1: mild, 2: moderate, 3: very_mild

    data = []
    result = []

    # Process each category
    categories = [
        (image_lists[0], 0),  # non_demented
        (image_lists[1], 1),  # mild_demented
        (image_lists[2], 2),  # moderate_demented
        (image_lists[3], 3),  # very_mild_demented
    ]

    for image_list, label in categories:
        logger.info(f"Processing {len(image_list)} images for label {label}")
        for img_path in tqdm(image_list):
            try:
                img = Image.open(img_path)
                img = img.resize(target_size)
                img = np.array(img)
                if img.shape == (target_size[0], target_size[1], 3):
                    data.append(np.array(img))
                    result.append(encoder.transform([[label]]).toarray())
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue

    X = np.array(data)
    y = np.array(result)
    y = y.reshape(X.shape[0], 4)
    y = np.argmax(y, axis=1)

    logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
    return X, y

def create_model(input_shape=(128, 128, 3)):
    """Create CNN model based on the notebook architecture"""
    model = Sequential()

    # First conv block
    model.add(Conv2D(filters=32, kernel_size=2, padding='Same', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=2, padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Second conv block
    model.add(Conv2D(filters=64, kernel_size=2, padding='Same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=2, padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Third conv block
    model.add(Conv2D(filters=128, kernel_size=2, padding='Same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=2, padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='softmax'))

    return model

def train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the model"""
    model = create_model()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.00001,
        patience=15,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=5,
    )

    checkpoint = ModelCheckpoint(
        'alz_model_new.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    logger.info("Evaluating model...")
    test_scores = model.evaluate(X_test, y_test, verbose=1)
    logger.info(f"Test Loss: {test_scores[0]:.4f}")
    logger.info(f"Test Accuracy: {test_scores[1]:.4f}")

    # Predictions for confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    logger.info("Confusion Matrix:")
    logger.info(cm)

    # Classification report
    class_names = ["Non Demented", "Mild Dementia", "Moderate Dementia", "Very mild Dementia"]
    report = classification_report(y_test, y_pred_classes, target_names=class_names)
    logger.info("Classification Report:")
    logger.info(report)

    return test_scores[1]  # Return accuracy

def save_model_info(accuracy, history):
    """Save model accuracy information"""
    model_info = {
        "alzheimer_cnn": {
            "validation_accuracy": float(accuracy),
            "model_path": "alz_model_new.h5",
            "classes": ["Non Demented", "Mild Dementia", "Moderate Dementia", "Very mild Dementia"],
            "input_shape": [128, 128, 3],
            "last_updated": str(tf.timestamp()),
            "model_type": "cnn_notebook_adapted",
            "training_status": "completed",
            "epochs_trained": len(history.history['accuracy']),
            "final_train_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1])
        }
    }

    # Load existing data
    try:
        with open("models/model_accuracy.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    # Update with new info
    data.update(model_info)

    # Save
    with open("models/model_accuracy.json", "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Model info saved with accuracy: {accuracy:.4f}")

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('alzheimer_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    data_dir = "dataset/alzheimer/Data"

    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found!")
        return

    # Load dataset
    logger.info("Loading dataset...")
    image_lists = load_dataset(data_dir)

    # Preprocess images
    logger.info("Preprocessing images...")
    X, y = preprocess_images(image_lists)

    # Split data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    logger.info(f"Training set: {X_train.shape}, {y_train.shape}")
    logger.info(f"Test set: {X_test.shape}, {y_test.shape}")

    # Train model
    model, history = train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)

    # Save model info
    save_model_info(accuracy, history)

    # Plot training history
    plot_training_history(history)

    logger.info(".4f")
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()