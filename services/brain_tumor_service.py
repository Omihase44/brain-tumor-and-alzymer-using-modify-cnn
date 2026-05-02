"""
Django Integration for Brain Tumor 4-Class CNN Classifier
API service for handling brain tumor classification predictions
"""

import json
import logging
from typing import Dict, Optional
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from models.brain_tumor_multiclass import get_multiclass_classifier

LOGGER = logging.getLogger(__name__)


class BrainTumorPredictionService:
    """Service for brain tumor predictions using the 4-class CNN model."""
    
    @staticmethod
    def predict_from_image_file(image_path: str) -> Dict[str, object]:
        """
        Predict tumor class from an image file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Prediction results dictionary
        """
        try:
            if not PIL_AVAILABLE:
                return {
                    "success": False,
                    "error": "PIL library not available"
                }
            
            # Load image
            img = Image.open(image_path)
            image_array = np.array(img)
            
            # Get classifier and predict
            classifier = get_multiclass_classifier()
            result = classifier.predict(image_array)
            
            return {
                "success": True,
                "prediction": result
            }
            
        except Exception as e:
            LOGGER.error(f"Error predicting from file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def predict_from_bytes(image_bytes: bytes) -> Dict[str, object]:
        """
        Predict tumor class from image bytes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Prediction results dictionary
        """
        try:
            if not PIL_AVAILABLE:
                return {
                    "success": False,
                    "error": "PIL library not available"
                }
            
            # Load image from bytes
            img = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(img)
            
            # Get classifier and predict
            classifier = get_multiclass_classifier()
            result = classifier.predict(image_array)
            
            return {
                "success": True,
                "prediction": result
            }
            
        except Exception as e:
            LOGGER.error(f"Error predicting from bytes: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def predict_from_array(image_array: np.ndarray) -> Dict[str, object]:
        """
        Predict tumor class from numpy array.
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Prediction results dictionary
        """
        try:
            classifier = get_multiclass_classifier()
            result = classifier.predict(image_array)
            
            return {
                "success": True,
                "prediction": result
            }
            
        except Exception as e:
            LOGGER.error(f"Error predicting from array: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def get_prediction_summary(prediction: Dict[str, object]) -> str:
        """
        Get a human-readable summary of the prediction.
        
        Args:
            prediction: Prediction result dictionary
            
        Returns:
            Summary string
        """
        if not prediction.get("success"):
            return f"Error: {prediction.get('error', 'Unknown error')}"
        
        pred = prediction.get("prediction", {})
        tumor_type = pred.get("tumor_type", "Unknown")
        confidence = pred.get("confidence", 0)
        grade = pred.get("grade", "Unknown")
        
        summary = f"{tumor_type} (Confidence: {confidence*100:.2f}%, Grade: {grade})"
        return summary
    
    @staticmethod
    def format_for_django(prediction: Dict[str, object]) -> Dict[str, object]:
        """
        Format prediction result for Django template context.
        
        Args:
            prediction: Prediction result dictionary
            
        Returns:
            Formatted dictionary for Django
        """
        if not prediction.get("success"):
            return {
                "error": prediction.get("error", "Unknown error"),
                "success": False
            }
        
        pred = prediction.get("prediction", {})
        
        # Extract model accuracy if available
        model_accuracy = pred.get("model_accuracy", {})
        
        return {
            "success": True,
            "tumor_detected": pred.get("detected", False),
            "tumor_type": pred.get("tumor_type", "Unknown"),
            "tumor_class": pred.get("tumor_class", "unknown"),
            "classification": pred.get("classification", "Unknown"),
            "grade": pred.get("grade", "Unknown"),
            "confidence": pred.get("confidence", 0),
            "confidence_percent": f"{pred.get('confidence', 0)*100:.2f}%",
            "scores": pred.get("scores", {}),
            "backend": pred.get("backend", "unknown"),
            "summary": BrainTumorPredictionService.get_prediction_summary(prediction),
            "model_accuracy": model_accuracy,
            "model_validation_accuracy": model_accuracy.get("validation_accuracy", 0),
            "model_test_accuracy": model_accuracy.get("test_accuracy", 0),
            "model_training_accuracy": model_accuracy.get("training_accuracy", 0)
        }


def predict_brain_tumor(image_path: Optional[str] = None, 
                       image_bytes: Optional[bytes] = None,
                       image_array: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Convenient function to make a brain tumor prediction.
    
    Args:
        image_path: Path to image file
        image_bytes: Image as bytes
        image_array: Image as numpy array
        
    Returns:
        Prediction results dictionary
    """
    service = BrainTumorPredictionService()
    
    if image_path:
        return service.predict_from_image_file(image_path)
    elif image_bytes:
        return service.predict_from_bytes(image_bytes)
    elif image_array is not None:
        return service.predict_from_array(image_array)
    else:
        return {
            "success": False,
            "error": "No image provided"
        }


if __name__ == "__main__":
    # Test the service
    logging.basicConfig(level=logging.INFO)
    
    service = BrainTumorPredictionService()
    LOGGER.info("Brain Tumor Prediction Service initialized")
