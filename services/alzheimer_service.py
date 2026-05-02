"""
Django Integration for Alzheimer Multiclass CNN Classifier
API service for handling Alzheimer disease stage classification predictions
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

from models.alzheimer_multiclass import AlzheimerMulticlassClassifier

LOGGER = logging.getLogger(__name__)


class AlzheimerPredictionService:
    """Service for Alzheimer disease predictions using the 4-class CNN model."""
    
    @staticmethod
    def predict_from_image_file(image_path: str) -> Dict[str, object]:
        """
        Predict Alzheimer stage from an image file path.
        
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
            
            # Get classifier and predict
            classifier = AlzheimerMulticlassClassifier()
            result = classifier.predict(image_path)
            
            if result:
                return {
                    "success": True,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "model_accuracy": result["model_accuracy"],
                    "all_probabilities": result["all_probabilities"]
                }
            else:
                return {
                    "success": False,
                    "error": "Prediction failed"
                }
                
        except Exception as e:
            LOGGER.error(f"Error in predict_from_image_file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def predict_from_bytes(image_bytes: bytes) -> Dict[str, object]:
        """
        Predict Alzheimer stage from image bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Prediction results dictionary
        """
        try:
            if not PIL_AVAILABLE:
                return {
                    "success": False,
                    "error": "PIL library not available"
                }
            
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(img)
            
            # For now, save to temp file since the classifier expects a path
            # TODO: Modify classifier to accept PIL images directly
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                img.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                result = AlzheimerPredictionService.predict_from_image_file(temp_path)
            finally:
                os.unlink(temp_path)
            
            return result
                
        except Exception as e:
            LOGGER.error(f"Error in predict_from_bytes: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def predict_from_array(image_array: np.ndarray) -> Dict[str, object]:
        """
        Predict Alzheimer stage from numpy array.
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Prediction results dictionary
        """
        try:
            if not PIL_AVAILABLE:
                return {
                    "success": False,
                    "error": "PIL library not available"
                }
            
            # Convert array to PIL Image
            img = Image.fromarray(image_array.astype('uint8'))
            
            # Save to temp file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                img.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                result = AlzheimerPredictionService.predict_from_image_file(temp_path)
            finally:
                os.unlink(temp_path)
            
            return result
                
        except Exception as e:
            LOGGER.error(f"Error in predict_from_array: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def predict_alzheimer(image_path: Optional[str] = None, 
                     image_bytes: Optional[bytes] = None, 
                     image_array: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Unified prediction function for Alzheimer disease classification.
    
    Args:
        image_path: Path to image file
        image_bytes: Raw image bytes  
        image_array: Image as numpy array
        
    Returns:
        Prediction results dictionary
    """
    if image_path:
        return AlzheimerPredictionService.predict_from_image_file(image_path)
    elif image_bytes:
        return AlzheimerPredictionService.predict_from_bytes(image_bytes)
    elif image_array is not None:
        return AlzheimerPredictionService.predict_from_array(image_array)
    else:
        return {
            "success": False,
            "error": "No image provided"
        }