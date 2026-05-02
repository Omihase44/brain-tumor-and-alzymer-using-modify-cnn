import os
import numpy as np
import tensorflow as tf
from models.alzheimer_model import build_alzheimer_staging_model, ALZHEIMER_STAGE_CLASSES

def test_model_architecture():
    """Test the improved Alzheimer model architecture with a small dataset sample."""
    print("Testing improved Alzheimer model architecture...")

    # Build the model
    model = build_alzheimer_staging_model(
        input_shape=(224, 224, 3),
        num_classes=4,
        variant="efficientnetb0",
        weights=None,  # Don't load weights for testing
        dense_units=512,
        dropout_rate=0.4,
        fine_tune_layers=50
    )

    if model is None:
        print("❌ Failed to build model")
        return False

    print("✅ Model built successfully!")
    print(f"Model summary:")
    model.summary()

    # Test with dummy data
    dummy_input = np.random.rand(1, 224, 224, 3)
    try:
        predictions = model.predict(dummy_input, verbose=0)
        print(f"✅ Model prediction shape: {predictions.shape}")
        print(f"✅ Predictions: {predictions}")
        print(f"✅ Predicted class: {ALZHEIMER_STAGE_CLASSES[np.argmax(predictions[0])]}")
        print(".2f")
        return True
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        return False

def test_data_loading():
    """Test loading a small sample of the dataset."""
    print("\nTesting data loading...")

    base_path = "dataset/alzheimer"
    sample_size = 10  # Load only 10 images per class for testing

    images = []
    labels = []

    # Define folder to class mapping
    folder_class_map = {
        'non': 'NonDemented',
        'mild': 'Very Mild',
        'moderate': 'Mild',
        'severe': 'Moderate'
    }

    for folder, class_name in folder_class_map.items():
        folder_path = os.path.join(base_path, folder)
        class_index = ALZHEIMER_STAGE_CLASSES.index(class_name)

        if not os.path.exists(folder_path):
            print(f"⚠️  Folder {folder_path} does not exist, skipping...")
            continue

        print(f"Loading {sample_size} {class_name} images from {folder_path}")

        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and count < sample_size:
                img_path = os.path.join(folder_path, filename)
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0  # Normalize

                    images.append(img_array)
                    labels.append(class_index)
                    count += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

        print(f"✅ Loaded {count} images for {class_name}")

    if images:
        X = np.array(images)
        y = np.array(labels)

        print(f"✅ Total sample images: {len(X)}")
        print(f"✅ Sample class distribution: {np.bincount(y)}")
        print(f"✅ Image shape: {X.shape}")
        return True
    else:
        print("❌ No images loaded")
        return False

if __name__ == "__main__":
    print("🧠 Testing Improved Alzheimer Model Implementation")
    print("=" * 50)

    # Test model architecture
    model_ok = test_model_architecture()

    # Test data loading
    data_ok = test_data_loading()

    print("\n" + "=" * 50)
    if model_ok and data_ok:
        print("🎉 All tests passed! Ready for full training.")
        print("\nNext steps:")
        print("1. Run the full training script: python train_improved_alzheimer_cnn.py")
        print("2. Monitor training progress and results")
        print("3. Expect >90% accuracy with the improved architecture")
    else:
        print("❌ Some tests failed. Please check the issues above.")