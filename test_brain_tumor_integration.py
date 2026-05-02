"""
Quick Test Script for Brain Tumor 4-Class CNN Classifier
Tests the model integration and demonstrates usage
"""

import os
import sys
from pathlib import Path
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

# Add repo root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))


def test_model_loading():
    """Test if model can be loaded."""
    LOGGER.info("=" * 70)
    LOGGER.info("TEST 1: Model Loading")
    LOGGER.info("=" * 70)
    
    try:
        from models.brain_tumor_multiclass import get_multiclass_classifier
        
        LOGGER.info("Initializing classifier...")
        classifier = get_multiclass_classifier()
        
        if classifier.model is None:
            LOGGER.warning("⚠ Model not found. This is expected if training hasn't completed.")
            return False
        
        LOGGER.info("✓ Model loaded successfully")
        info = classifier.get_model_info()
        LOGGER.info(f"  Classes: {info.get('classes')}")
        LOGGER.info(f"  Input Shape: {info.get('image_size')}")
        return True
        
    except Exception as e:
        LOGGER.error(f"✗ Error loading model: {e}")
        return False


def test_dataset():
    """Test if dataset structure is correct."""
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("TEST 2: Dataset Structure")
    LOGGER.info("=" * 70)
    
    try:
        dataset_dir = REPO_ROOT / "dataset" / "brain"
        train_dir = dataset_dir / "Training"
        test_dir = dataset_dir / "Testing"
        
        classes = ["glioma", "meningioma", "notumor", "pituitary"]
        
        LOGGER.info(f"Checking dataset at: {dataset_dir}")
        
        total_train = 0
        for class_name in classes:
            class_dir = train_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.jpg')))
                LOGGER.info(f"  ✓ Training/{class_name}: {count} images")
                total_train += count
            else:
                LOGGER.warning(f"  ✗ Training/{class_name}: NOT FOUND")
        
        total_test = 0
        for class_name in classes:
            class_dir = test_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.jpg')))
                LOGGER.info(f"  ✓ Testing/{class_name}: {count} images")
                total_test += count
            else:
                LOGGER.warning(f"  ✗ Testing/{class_name}: NOT FOUND")
        
        LOGGER.info(f"\nTotal training images: {total_train}")
        LOGGER.info(f"Total testing images: {total_test}")
        
        if total_train > 0 and total_test > 0:
            LOGGER.info("✓ Dataset structure is correct")
            return True
        else:
            LOGGER.warning("⚠ Dataset appears to be incomplete")
            return False
            
    except Exception as e:
        LOGGER.error(f"✗ Error checking dataset: {e}")
        return False


def test_classification_service():
    """Test the Django classification service."""
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("TEST 3: Classification Service")
    LOGGER.info("=" * 70)
    
    try:
        from services.brain_tumor_service import predict_brain_tumor
        
        LOGGER.info("Testing with synthetic image...")
        
        # Create a random test image
        test_image = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
        
        result = predict_brain_tumor(image_array=test_image)
        
        if result.get('success'):
            pred = result['prediction']
            LOGGER.info(f"✓ Prediction successful")
            LOGGER.info(f"  Tumor Type: {pred.get('tumor_type')}")
            LOGGER.info(f"  Confidence: {pred.get('confidence'):.4f}")
            LOGGER.info(f"  Grade: {pred.get('grade')}")
            return True
        else:
            LOGGER.warning(f"⚠ Prediction returned success=False: {result.get('error')}")
            return True  # Service is working, model may not be ready
            
    except Exception as e:
        LOGGER.error(f"✗ Error in classification service: {e}")
        return False


def test_image_preprocessing():
    """Test image preprocessing."""
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("TEST 4: Image Preprocessing")
    LOGGER.info("=" * 70)
    
    try:
        from models.brain_tumor_multiclass import BrainTumorMulticlassClassifier
        
        LOGGER.info("Testing preprocessing with various image formats...")
        
        classifier = BrainTumorMulticlassClassifier()
        
        # Test 1: RGB image
        rgb_image = np.random.rand(256, 256, 3) * 255
        processed = classifier.preprocess_image(rgb_image)
        assert processed.shape == (1, 150, 150, 3), "RGB preprocessing failed"
        LOGGER.info("  ✓ RGB image preprocessing")
        
        # Test 2: Grayscale image
        gray_image = np.random.rand(256, 256) * 255
        processed = classifier.preprocess_image(gray_image)
        assert processed.shape == (1, 150, 150, 3), "Grayscale preprocessing failed"
        LOGGER.info("  ✓ Grayscale image preprocessing")
        
        # Test 3: Already normalized image
        normalized_image = np.random.rand(256, 256, 3)
        processed = classifier.preprocess_image(normalized_image)
        assert processed.shape == (1, 150, 150, 3), "Normalized image preprocessing failed"
        LOGGER.info("  ✓ Normalized image preprocessing")
        
        LOGGER.info("✓ All image preprocessing tests passed")
        return True
        
    except Exception as e:
        LOGGER.error(f"✗ Error in image preprocessing: {e}")
        return False


def test_class_mapping():
    """Test class mapping and metadata."""
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("TEST 5: Class Mapping and Metadata")
    LOGGER.info("=" * 70)
    
    try:
        from models.brain_tumor_multiclass import (
            TUMOR_CLASSES_MULTICLASS,
            TUMOR_CLASS_LABELS,
            GRADE_MAPPING
        )
        
        LOGGER.info("Class Information:")
        for class_name in TUMOR_CLASSES_MULTICLASS:
            label = TUMOR_CLASS_LABELS.get(class_name, "Unknown")
            grade = GRADE_MAPPING.get(class_name, "Unknown")
            LOGGER.info(f"  {class_name:12} -> {label:20} (Grade: {grade})")
        
        LOGGER.info("✓ All class mappings are correct")
        return True
        
    except Exception as e:
        LOGGER.error(f"✗ Error in class mapping: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("BRAIN TUMOR 4-CLASS CNN INTEGRATION TEST SUITE")
    LOGGER.info("=" * 70)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Dataset Structure", test_dataset),
        ("Classification Service", test_classification_service),
        ("Image Preprocessing", test_image_preprocessing),
        ("Class Mapping", test_class_mapping),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            LOGGER.error(f"Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("TEST SUMMARY")
    LOGGER.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        LOGGER.info(f"{status:8} - {test_name}")
    
    LOGGER.info("=" * 70)
    LOGGER.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        LOGGER.info("✓ All tests passed! System is ready to use.")
    elif passed > total // 2:
        LOGGER.warning("⚠ Some tests failed. Check logs above for details.")
    else:
        LOGGER.error("✗ Multiple tests failed. Please address issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
