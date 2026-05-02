"""
Brain Tumor Classification CNN Model Training Script (4-Class)
Based on: CNN Brain Tumor Classification 99% Accuracy Notebook
Classes: Glioma, Meningioma, No Tumor, Pituitary
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
import json
from pathlib import Path

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 40
CLASS_TYPES = ["glioma", "meningioma", "notumor", "pituitary"]
N_TYPES = 4

# Dataset paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "brain"
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"
MODELS_DIR = BASE_DIR / "trained_models"
MODELS_DIR.mkdir(exist_ok=True)

# Output model path
TUMOR_MODEL_PATH = MODELS_DIR / "brain_tumor_cnn_multiclass.h5"
TUMOR_MODEL_METADATA = MODELS_DIR / "tumor_model_multiclass_metadata.json"


def load_data_generators():
    """Create data generators for training and validation."""
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        brightness_range=(0.85, 1.15),
        width_shift_range=0.002,
        height_shift_range=0.002,
        shear_range=12.5,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest"
    )
    
    # Validation data (no augmentation, just rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED
    )
    
    test_generator = test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=SEED
    )
    
    return train_generator, test_generator


def build_tumor_model():
    """Build the CNN model architecture based on notebook."""
    
    image_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    model = models.Sequential([
        # Convolutional layer 1
        layers.Conv2D(32, (4, 4), activation="relu", input_shape=image_shape),
        layers.MaxPooling2D(pool_size=(3, 3)),
        
        # Convolutional layer 2
        layers.Conv2D(64, (4, 4), activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3)),
        
        # Convolutional layer 3
        layers.Conv2D(128, (4, 4), activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3)),
        
        # Convolutional layer 4
        layers.Conv2D(128, (4, 4), activation="relu"),
        layers.Flatten(),
        
        # Fully connected layers
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5, seed=SEED),
        layers.Dense(N_TYPES, activation="softmax")
    ])
    
    # Compile with optimized Adam parameters
    optimizer = Adam(learning_rate=0.001, beta_1=0.869, beta_2=0.995)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, train_generator, test_generator):
    """Train the model."""
    
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = test_generator.samples // BATCH_SIZE
    
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Image shape: {IMAGE_SIZE + (3,)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Total training samples: {train_generator.samples}")
    print(f"Total validation samples: {test_generator.samples}")
    print(f"{'='*70}\n")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='loss',
        min_delta=1e-9,
        patience=8,
        verbose=True,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        verbose=True
    )
    
    # Train
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr]
    )
    
    return history


def evaluate_model(model, test_generator):
    """Evaluate model on test set."""
    
    batch_size = test_generator.batch_size
    steps = test_generator.samples // batch_size
    
    loss, accuracy = model.evaluate(test_generator, steps=steps)
    
    print(f"\n{'='*70}")
    print("MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Test Loss: {loss:.5f}")
    print(f"Test Accuracy: {accuracy:.5f}")
    print(f"{'='*70}\n")
    
    return loss, accuracy


def save_model(model, metadata):
    """Save the model and metadata."""
    
    # Save model
    model.save(str(TUMOR_MODEL_PATH))
    print(f"\n✓ Model saved to: {TUMOR_MODEL_PATH}")
    
    # Save metadata
    with open(TUMOR_MODEL_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {TUMOR_MODEL_METADATA}")
    
    return TUMOR_MODEL_PATH


def get_class_distribution(generator):
    """Get class distribution from generator."""
    
    class_indices = generator.class_indices
    classes = sorted(class_indices.items(), key=lambda x: x[1])
    
    distribution = {}
    for class_name, class_idx in classes:
        count = np.sum(generator.classes == class_idx)
        distribution[class_name] = int(count)
    
    return distribution


def main():
    """Main training function."""
    
    print("\n" + "="*70)
    print("BRAIN TUMOR CLASSIFICATION CNN MODEL TRAINING (4-CLASS)")
    print("="*70 + "\n")
    
    # Check dataset exists
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(f"Dataset not found. Expected:\n  {TRAIN_DIR}\n  {TEST_DIR}")
    
    print(f"✓ Dataset found")
    print(f"  Train dir: {TRAIN_DIR}")
    print(f"  Test dir:  {TEST_DIR}")
    
    # Load data
    print("\nLoading data generators...")
    train_gen, test_gen = load_data_generators()
    print(f"✓ Data generators created")
    
    # Get class distributions
    train_dist = get_class_distribution(train_gen)
    test_dist = get_class_distribution(test_gen)
    
    print(f"\n  Training distribution: {train_dist}")
    print(f"  Testing distribution: {test_dist}")
    
    # Build model
    print("\nBuilding model...")
    model = build_tumor_model()
    print(f"✓ Model built")
    print(f"\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nStarting training...")
    history = train_model(model, train_gen, test_gen)
    
    # Evaluate
    loss, accuracy = evaluate_model(model, test_gen)
    
    # Save model
    print("Saving model...")
    metadata = {
        "model_name": "Brain Tumor CNN (4-Class)",
        "classes": CLASS_TYPES,
        "image_size": IMAGE_SIZE,
        "test_accuracy": float(accuracy),
        "test_loss": float(loss),
        "training_samples": train_gen.samples,
        "validation_samples": test_gen.samples,
        "batch_size": BATCH_SIZE,
        "epochs_trained": len(history.history['loss']),
        "class_distribution": {
            "training": train_dist,
            "testing": test_dist
        }
    }
    
    save_model(model, metadata)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel path: {TUMOR_MODEL_PATH}")
    print(f"Metadata path: {TUMOR_MODEL_METADATA}")
    
    return model


if __name__ == "__main__":
    try:
        model = main()
        print("\n✓ Ready for deployment")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
