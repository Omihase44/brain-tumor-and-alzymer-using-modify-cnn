"""
Brain Tumor 4-Class CNN Classifier (Glioma, Meningioma, No Tumor, Pituitary)
Integration module for the trained CNN model
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# Class definitions for 4-class tumor classification
TUMOR_CLASSES_MULTICLASS = ["glioma", "meningioma", "notumor", "pituitary"]
TUMOR_CLASS_LABELS = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor"
}

# Grade mapping for different tumor types
GRADE_MAPPING = {
    "glioma": "Grade III",
    "meningioma": "Grade II",
    "pituitary": "Grade IV",
    "notumor": "None"
}

# Get model paths
REPO_ROOT = Path(__file__).parent.parent
TRAINED_MODELS_DIR = REPO_ROOT / "trained_models"
MODEL_PATH = TRAINED_MODELS_DIR / "brain_tumor_cnn_multiclass_improved.h5"
METADATA_PATH = TRAINED_MODELS_DIR / "tumor_model_multiclass_improved_metadata.json"


class BrainTumorMulticlassClassifier:
    """
    4-Class Brain Tumor CNN Classifier
    Classifies brain MRI images into: Glioma, Meningioma, No Tumor, or Pituitary
    """
    
    def __init__(self, model_path: str = None, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the classifier with a trained model.
        
        Args:
            model_path: Path to the trained H5 model file
            image_size: Target image size for model input
        """
        self.image_size = image_size
        self.model = None
        self.model_path = model_path or str(MODEL_PATH)
        self.classes = TUMOR_CLASSES_MULTICLASS
        self.class_labels = TUMOR_CLASS_LABELS
        self.grade_mapping = GRADE_MAPPING
        
        if TENSORFLOW_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                LOGGER.info(f"Loading model from {self.model_path}")
                self.model = load_model(self.model_path)
                LOGGER.info("Model loaded successfully")
            else:
                LOGGER.warning(f"Model not found at {self.model_path}")
        except Exception as e:
            LOGGER.error(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model prediction.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for model
        """
        # Resize image
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        
        # Resize to model input size
        if image.shape[:2] != self.image_size:
            # Using PIL for resizing
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray((image * 255).astype(np.uint8) if image.dtype == np.float32 else image.astype(np.uint8))
            pil_img = pil_img.resize(self.image_size, PILImage.LANCZOS)
            image = np.array(pil_img)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1:
            image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image: np.ndarray) -> Dict[str, object]:
        """
        Predict tumor class for the given image.
        
        Args:
            image: Input brain MRI image as numpy array
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            LOGGER.warning("Model not available for prediction")
            return {
                "detected": False,
                "classification": "Unknown",
                "tumor_type": "Unknown",
                "grade": "Unknown",
                "confidence": 0.0,
                "error": "Model not available"
            }
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get predictions
            predictions = self.model.predict(processed_image, verbose=0)
            scores = predictions[0]
            
            # Get predicted class
            predicted_idx = np.argmax(scores)
            predicted_class = self.classes[predicted_idx]
            confidence = float(scores[predicted_idx])
            
            # Determine if tumor is detected
            tumor_detected = predicted_class != "notumor"
            
            # Get grade
            grade = self.grade_mapping.get(predicted_class, "Unknown")
            
            # Get label - show actual tumor type when detected
            if tumor_detected:
                label = self.class_labels.get(predicted_class, predicted_class)
                display_tumor_type = label
                display_grade = grade
            else:
                label = self.class_labels.get(predicted_class, predicted_class)
                display_tumor_type = label
                display_grade = grade
            
            # Load model accuracy info
            model_accuracy = self._load_accuracy_info()
            
            result = {
                "detected": tumor_detected,
                "classification": label,
                "tumor_type": display_tumor_type,
                "tumor_class": predicted_class,  # Keep original class for internal use
                "grade": display_grade,
                "confidence": round(confidence, 4),
                "scores": {
                    self.classes[i]: float(scores[i]) 
                    for i in range(len(self.classes))
                },
                "backend": "keras_multiclass",
                "model_accuracy": model_accuracy
            }
            
            LOGGER.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            LOGGER.error(f"Error during prediction: {e}")
            return {
                "detected": False,
                "classification": "Error",
                "tumor_type": "Error",
                "grade": "Unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def predict_batch(self, images: np.ndarray) -> list:
        """
        Predict tumor class for a batch of images.
        
        Args:
            images: Batch of images as numpy array
            
        Returns:
            List of prediction results
        """
        results = []
        for i in range(images.shape[0]):
            result = self.predict(images[i])
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, object]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        # Load accuracy information from metadata
        accuracy_info = self._load_accuracy_info()
        
        return {
            "status": "loaded",
            "classes": self.classes,
            "class_labels": self.class_labels,
            "model_path": self.model_path,
            "image_size": self.image_size,
            "model_summary": str(self.model.summary()) if self.model else None,
            "accuracy": accuracy_info
        }
    
    def _load_accuracy_info(self) -> Dict[str, object]:
        """Load accuracy information from metadata file."""
        try:
            # Try multiple possible metadata file locations
            possible_paths = [
                self.model_path.replace('.h5', '_metadata.json'),
                os.path.join(os.path.dirname(self.model_path), 'tumor_model_multiclass_improved_metadata.json'),
                os.path.join(os.path.dirname(self.model_path), 'tumor_model_multiclass_metadata.json'),
                str(METADATA_PATH)
            ]
            
            for metadata_file in possible_paths:
                if os.path.exists(metadata_file):
                    LOGGER.info(f"Loading metadata from {metadata_file}")
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    return {
                        "validation_accuracy": metadata.get("best_val_accuracy", 0),
                        "test_accuracy": metadata.get("final_test_accuracy", 0),
                        "training_accuracy": metadata.get("final_train_accuracy", 0),
                        "epochs_trained": metadata.get("epochs_trained", 0),
                        "best_epoch": metadata.get("best_epoch", 0),
                        "improvements": metadata.get("improvements", [])
                    }
            
            LOGGER.warning(f"No metadata file found in: {possible_paths}")
            return {}
            
        except Exception as e:
            LOGGER.warning(f"Could not load accuracy metadata: {e}")
            return {}


# Global classifier instance
_classifier_instance = None


def get_multiclass_classifier() -> BrainTumorMulticlassClassifier:
    """Get or create the global classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = BrainTumorMulticlassClassifier()
    return _classifier_instance


def predict_tumor_multiclass(image: np.ndarray) -> Dict[str, object]:
    """
    Convenience function to predict tumor class.
    
    Args:
        image: Brain MRI image as numpy array
        
    Returns:
        Prediction results
    """
    classifier = get_multiclass_classifier()
    return classifier.predict(image)


if __name__ == "__main__":
    # Test the classifier
    logging.basicConfig(level=logging.INFO)
    
    classifier = BrainTumorMulticlassClassifier()
    info = classifier.get_model_info()
    print("Classifier Info:", info)

