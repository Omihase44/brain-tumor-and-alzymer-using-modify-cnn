import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import the improved model
from models.alzheimer_model import build_alzheimer_staging_model, ALZHEIMER_STAGE_CLASSES

def create_efficient_dataset(base_path="dataset/alzheimer", batch_size=32, img_size=(224, 224)):
    """
    Create efficient tf.data.Dataset from consolidated Alzheimer folders.
    """
    # Define folder to class mapping
    folder_class_map = {
        'non': 0,      # NonDemented
        'mild': 1,     # Very Mild
        'moderate': 2, # Mild
        'severe': 3    # Moderate
    }

    all_file_paths = []
    all_labels = []

    # Collect all file paths and labels
    for folder, class_index in folder_class_map.items():
        folder_path = os.path.join(base_path, folder)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue

        print(f"Scanning {folder_path} for {ALZHEIMER_STAGE_CLASSES[class_index]} images...")

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(folder_path, filename)
                all_file_paths.append(file_path)
                all_labels.append(class_index)

    print(f"Total images found: {len(all_file_paths)}")

    # Convert to numpy arrays
    file_paths = np.array(all_file_paths)
    labels = np.array(all_labels)

    # Split into train and validation with strict separation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels,
        test_size=0.2,  # 80:20 split
        stratify=labels,  # Maintain class distribution
        random_state=42,  # Fixed seed for reproducibility
        shuffle=True  # Ensure proper shuffling
    )

    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Class distribution - Train: {np.bincount(train_labels)}")
    print(f"Class distribution - Val: {np.bincount(val_labels)}")

    # Create datasets
    train_ds = create_tf_dataset(train_paths, train_labels, batch_size, img_size, training=True)
    val_ds = create_tf_dataset(val_paths, val_labels, batch_size, img_size, training=False)

    return train_ds, val_ds, val_paths, val_labels, train_labels

def create_tf_dataset(file_paths, labels, batch_size, img_size, training=True):
    """
    Create optimized tf.data.Dataset from file paths and labels.
    """

    def load_and_preprocess_image(file_path, label):
        # Load image
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    def augment_image(image, label):
        # Data augmentation for training only (as per requirements)
        # width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True
        # Note: rotation_range=20 removed due to TensorFlow version compatibility
        image = tf.image.random_flip_left_right(image)  # horizontal_flip=True

        # Simulate width_shift_range and height_shift_range with random crop and resize
        crop_size = tf.random.uniform([], 200, 224, dtype=tf.int32)  # Between 200-224 (shift_range=0.1)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
        image = tf.image.resize(image, (224, 224))  # Resize back to original

        # Simulate zoom_range with random scaling
        scale_factor = tf.random.uniform([], 0.9, 1.1)  # zoom_range=0.1
        new_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * scale_factor, tf.int32)
        image = tf.image.resize(image, new_size)
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)  # Center crop/pad back

        return image, label

    # Create dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Load and preprocess images
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        # Apply data augmentation
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle, batch, and prefetch
    if training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def train_improved_alzheimer_model_efficient():
    """
    Train the improved Alzheimer staging model using efficient tf.data.Dataset.
    """
    print("🚀 Starting efficient improved Alzheimer model training...")

    # Create efficient datasets
    print("Creating efficient datasets...")
    train_ds, val_ds, val_paths, val_labels, train_labels = create_efficient_dataset(batch_size=16)  # Smaller batch size for stability

    # Build improved model
    print("🏗️  Building improved model with EfficientNetB0...")
    model = build_alzheimer_staging_model(
        input_shape=(224, 224, 3),
        num_classes=4,
        variant="efficientnetb0",
        weights="imagenet",
        dense_units=512,
        dropout_rate=0.4,
        fine_tune_layers=50
    )

    if model is None:
        print("❌ Failed to build model. TensorFlow/Keras not available.")
        return

    # Compile model with optimized settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Use sparse for integer labels
        metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
    )

    # Compute class weights for imbalanced dataset
    print("⚖️  Computing class weights for imbalanced dataset...")
    # Get all training labels by iterating through the dataset
    train_labels_list = []
    for _, labels_batch in train_ds:
        train_labels_list.extend(labels_batch.numpy())
    train_labels_array = np.array(train_labels_list)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels_array),
        y=train_labels_array
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Callbacks for training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Changed to val_loss as requested
            patience=5,          # Reduced patience to 5
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'alz_model_improved_efficient.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log_improved.csv')
    ]

    # Calculate steps
    train_steps = len(train_ds)
    val_steps = len(val_ds)

    print(f"📊 Training with {train_steps} steps per epoch")
    print(f"📊 Validation with {val_steps} steps per epoch")

    # Train the model
    print("🎯 Starting training with class weights...")
    history = model.fit(
        train_ds,
        epochs=30,  # Reasonable number of epochs
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Apply class weights
        verbose=1
    )

    # Load best model
    print("🔄 Loading best model...")
    model = tf.keras.models.load_model('alz_model_improved_efficient.h5')

    # Evaluate the model
    print("📈 Evaluating model...")
    val_loss, val_accuracy, val_acc = model.evaluate(val_ds, verbose=1)

    # Generate predictions for detailed metrics
    print("🔍 Generating detailed predictions...")
    y_pred = model.predict(val_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(val_labels, y_pred_classes, target_names=ALZHEIMER_STAGE_CLASSES))

    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(val_labels, y_pred_classes)
    print(cm)

    # Save training history and metrics
    training_results = {
        'timestamp': datetime.now().isoformat(),
        'model_variant': 'efficientnetb0_efficient',
        'final_accuracy': float(val_accuracy),
        'final_loss': float(val_loss),
        'epochs_trained': len(history.history['accuracy']),
        'class_names': ALZHEIMER_STAGE_CLASSES,
        'training_history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(val_labels, y_pred_classes,
                                                     target_names=ALZHEIMER_STAGE_CLASSES,
                                                     output_dict=True)
    }

    # Save results
    with open('models/model_accuracy_improved_efficient.json', 'w') as f:
        json.dump(training_results, f, indent=2)

    # Also update the main model_accuracy.json
    try:
        with open('models/model_accuracy.json', 'r') as f:
            main_accuracy_data = json.load(f)
    except FileNotFoundError:
        main_accuracy_data = {}

    # Update the alzheimer_multiclass entry with new accuracy
    main_accuracy_data['alzheimer_multiclass'] = {
        "validation_accuracy": float(val_accuracy),
        "accuracy": float(val_accuracy),  # Add accuracy field
        "model_path": "alz_model_improved_efficient.h5",
        "classes": ALZHEIMER_STAGE_CLASSES,
        "input_shape": [224, 224, 3],
        "last_updated": datetime.now().isoformat(),
        "model_type": "EfficientNetB0_transfer_learning",
        "training_status": "completed",
        "epochs_completed": len(history.history['accuracy']),
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1])
    }

    with open('models/model_accuracy.json', 'w') as f:
        json.dump(main_accuracy_data, f, indent=4)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.bar(ALZHEIMER_STAGE_CLASSES, np.bincount(val_labels), alpha=0.7, label='True')
    plt.bar(ALZHEIMER_STAGE_CLASSES, np.bincount(y_pred_classes), alpha=0.7, label='Predicted')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('training_curves_improved_efficient.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n📊 Final Results:")
    print(f"Final Validation Accuracy: {val_accuracy:.2%}")
    print(f"Final Validation Loss: {val_loss:.4f}")
    if val_accuracy >= 0.90:
        print("🎉 SUCCESS: Achieved target accuracy of 90%+!")
    elif val_accuracy >= 0.85:
        print("👍 GOOD: Achieved accuracy above 85%. Close to target!")
    else:
        print(f"⚠️  Accuracy below target. Current: {val_accuracy:.2%}, Target: 90%+")

    return model, training_results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Train the improved model
    model, results = train_improved_alzheimer_model_efficient()

    print("\n✅ Training completed!")
    print("📁 Model saved as: alz_model_improved_efficient.h5")
    print("📄 Results saved to: models/model_accuracy_improved_efficient.json")
    print("📊 Training curves saved as: training_curves_improved_efficient.png")
    print("📝 Training log saved as: training_log_improved.csv")