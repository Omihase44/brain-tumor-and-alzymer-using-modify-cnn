import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import the improved model
from models.alzheimer_model import build_alzheimer_staging_model, ALZHEIMER_STAGE_CLASSES

def load_and_consolidate_dataset(base_path="dataset/alzheimer", target_size=(224, 224), batch_size=32):
    """
    Load and consolidate Alzheimer dataset from multiple folders.
    Maps folder names to standard class names.
    """
    # Define folder to class mapping
    folder_class_map = {
        'non': 'NonDemented',      # No Impairment
        'mild': 'Very Mild',       # Very Mild Impairment
        'moderate': 'Mild',        # Mild Impairment
        'severe': 'Moderate'       # Moderate Impairment
    }

    # Initialize data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create consolidated train and validation generators
    train_generator = None
    validation_generator = None

    for folder, class_name in folder_class_map.items():
        folder_path = os.path.join(base_path, folder)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue

        print(f"Loading data from {folder_path} -> {class_name}")

        # Create temporary generator for this folder
        temp_train_gen = train_datagen.flow_from_directory(
            base_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=[folder],  # Only load this specific folder
            subset='training',
            shuffle=True
        )

        temp_val_gen = train_datagen.flow_from_directory(
            base_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=[folder],  # Only load this specific folder
            subset='validation',
            shuffle=False
        )

        # Remap class indices to our standard classes
        class_indices = {folder: ALZHEIMER_STAGE_CLASSES.index(class_name)}

        # Update generators with correct class mapping
        if train_generator is None:
            train_generator = temp_train_gen
            validation_generator = temp_val_gen
            train_generator.class_indices = class_indices
            validation_generator.class_indices = class_indices
        else:
            # Combine generators (this is a simplified approach)
            # In practice, you might want to create a custom generator
            pass

    # For simplicity, let's use a different approach - load all images manually
    return load_images_manually(base_path, folder_class_map, target_size, batch_size)

def load_images_manually(base_path, folder_class_map, target_size=(224, 224), batch_size=32, test_split=0.2):
    """
    Manually load and preprocess all images from the consolidated dataset.
    """
    images = []
    labels = []

    for folder, class_name in folder_class_map.items():
        folder_path = os.path.join(base_path, folder)
        class_index = ALZHEIMER_STAGE_CLASSES.index(class_name)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue

        print(f"Loading {class_name} images from {folder_path}")

        # Load all images from this folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0  # Normalize

                    images.append(img_array)
                    labels.append(class_index)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    print(f"Total images loaded: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=42
    )

    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(ALZHEIMER_STAGE_CLASSES))
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(ALZHEIMER_STAGE_CLASSES))

    # Create data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=batch_size, shuffle=True)

    # Create test generator without augmentation
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow(X_test, y_test_cat, batch_size=batch_size, shuffle=False)

    return train_generator, test_generator, X_test, y_test

def train_improved_alzheimer_model():
    """
    Train the improved Alzheimer staging model with high accuracy target.
    """
    print("Starting improved Alzheimer model training...")

    # Load consolidated dataset
    print("Loading and consolidating dataset...")
    train_generator, test_generator, X_test, y_test = load_and_consolidate_dataset()

    # Build improved model
    print("Building improved model with EfficientNetB0...")
    model = build_alzheimer_staging_model(
        input_shape=(224, 224, 3),
        num_classes=4,
        variant="efficientnetb0",  # Use EfficientNetB0 for better performance
        weights="imagenet",
        dense_units=512,
        dropout_rate=0.4,
        fine_tune_layers=50  # Fine-tune last 50 layers
    )

    if model is None:
        print("Failed to build model. TensorFlow/Keras not available.")
        return

    # Compile model with optimized settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Callbacks for training
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            'alz_model_improved.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Calculate steps
    steps_per_epoch = len(train_generator)
    validation_steps = len(test_generator)

    print(f"Training with {steps_per_epoch} steps per epoch")
    print(f"Validation with {validation_steps} steps")

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=test_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy, test_auc = model.evaluate(test_generator, verbose=1)

    # Generate predictions for detailed metrics
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=ALZHEIMER_STAGE_CLASSES))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)

    # Save training history and metrics
    training_results = {
        'timestamp': datetime.now().isoformat(),
        'model_variant': 'efficientnetb0',
        'final_accuracy': float(test_accuracy),
        'final_auc': float(test_auc),
        'final_loss': float(test_loss),
        'epochs_trained': len(history.history['accuracy']),
        'class_names': ALZHEIMER_STAGE_CLASSES,
        'training_history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'auc': [float(x) for x in history.history['auc']],
            'val_auc': [float(x) for x in history.history['val_auc']]
        }
    }

    # Save results
    with open('models/model_accuracy_improved.json', 'w') as f:
        json.dump(training_results, f, indent=2)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(".2f")
    print(".2f")
    print(".2f")

    if test_accuracy >= 0.90:
        print("🎉 SUCCESS: Achieved target accuracy of 90%+!")
    else:
        print(f"⚠️  Accuracy below target. Current: {test_accuracy:.2%}, Target: 90%+")

    return model, training_results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Train the improved model
    model, results = train_improved_alzheimer_model()

    print("\nTraining completed!")
    print("Model saved as: alz_model_improved.h5")
    print("Results saved to: models/model_accuracy_improved.json")
    print("Training curves saved as: training_curves_improved.png")