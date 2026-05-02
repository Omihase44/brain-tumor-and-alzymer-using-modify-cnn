"""
Model Image Routing & Validation Service
Ensures correct model routing and prevents cross-disease misclassification
"""

import logging
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
import os

LOGGER = logging.getLogger(__name__)

class ImageValidator:
    """Validates and routes images to correct models"""
    
    # Brain Tumor expected characteristics
    BRAIN_TUMOR_SPECS = {
        "target_size": (150, 150),
        "color_mode": "RGB",
        "expected_channels": 3,
        "model_name": "Brain Tumor CNN (4-Class)",
        "classes": ["glioma", "meningioma", "notumor", "pituitary"]
    }
    
    # Alzheimer expected characteristics
    ALZHEIMER_SPECS = {
        "target_size": (224, 224),
        "color_mode": "RGB",
        "expected_channels": 3,
        "model_name": "Alzheimer EfficientNetB0",
        "classes": ["NonDemented", "Very Mild", "Mild", "Moderate"]
    }
    
    @staticmethod
    def validate_image_file(image_path: str) -> Tuple[bool, str]:
        """
        Validate image file exists and is accessible
        
        Args:
            image_path: Path to image file
            
        Returns:
            (is_valid, message)
        """
        if not os.path.exists(image_path):
            return False, f"Image file not found: {image_path}"
        
        try:
            img = Image.open(image_path)
            img.verify()
            return True, "Image valid"
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    @staticmethod
    def preprocess_brain_tumor_image(image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image specifically for Brain Tumor model (150x150)
        
        Args:
            image_path: Path to brain MRI image
            
        Returns:
            Preprocessed numpy array or None if error
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize to brain tumor model size
            img = img.resize(ImageValidator.BRAIN_TUMOR_SPECS["target_size"], Image.LANCZOS)
            
            # Convert to array
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1]
            if img_array.max() > 1:
                img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            LOGGER.info(f"Brain Tumor image preprocessed: shape={img_array.shape}")
            return img_array
            
        except Exception as e:
            LOGGER.error(f"Error preprocessing brain tumor image: {e}")
            return None
    
    @staticmethod
    def preprocess_alzheimer_image(image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image specifically for Alzheimer model (224x224)
        
        Args:
            image_path: Path to brain MRI image
            
        Returns:
            Preprocessed numpy array or None if error
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize to Alzheimer model size
            img = img.resize(ImageValidator.ALZHEIMER_SPECS["target_size"], Image.LANCZOS)
            
            # Convert to array
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1]
            if img_array.max() > 1:
                img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            LOGGER.info(f"Alzheimer image preprocessed: shape={img_array.shape}")
            return img_array
            
        except Exception as e:
            LOGGER.error(f"Error preprocessing Alzheimer image: {e}")
            return None
    
    @staticmethod
    def validate_prediction_output(predictions: Dict, model_type: str) -> Dict:
        """
        Validate and sanitize model predictions
        
        Args:
            predictions: Raw model predictions
            model_type: Either 'brain_tumor' or 'alzheimer'
            
        Returns:
            Validated prediction dictionary
        """
        if model_type == "brain_tumor":
            specs = ImageValidator.BRAIN_TUMOR_SPECS
            expected_classes = len(specs["classes"])
        else:
            specs = ImageValidator.ALZHEIMER_SPECS
            expected_classes = len(specs["classes"])
        
        # Ensure predictions have correct structure
        if not isinstance(predictions, dict):
            LOGGER.error(f"Invalid predictions type for {model_type}")
            return {"error": f"Invalid predictions for {model_type}"}
        
        # Validate class is in expected list
        pred_class = predictions.get("classification", "unknown")
        if model_type == "brain_tumor" and pred_class not in specs["classes"]:
            LOGGER.warning(f"Unexpected class for brain tumor: {pred_class}")
        
        if model_type == "alzheimer" and pred_class not in specs["classes"]:
            LOGGER.warning(f"Unexpected class for Alzheimer: {pred_class}")
        
        return predictions


def separate_and_predict(image_path: str, model_type: str):
    """
    Main function to route image to correct model with proper preprocessing
    
    Args:
        image_path: Path to uploaded image
        model_type: 'brain_tumor' or 'alzheimer'
        
    Returns:
        Prediction result
    """
    # Validate image exists
    is_valid, msg = ImageValidator.validate_image_file(image_path)
    if not is_valid:
        LOGGER.error(f"Image validation failed: {msg}")
        return {"success": False, "error": msg}
    
    LOGGER.info(f"Processing {model_type} image: {image_path}")
    
    if model_type == "brain_tumor":
        # Preprocess for brain tumor
        preprocessed = ImageValidator.preprocess_brain_tumor_image(image_path)
        if preprocessed is None:
            return {"success": False, "error": "Failed to preprocess image for brain tumor model"}
        
        # Import and use brain tumor model
        try:
            from models.brain_tumor_multiclass import BrainTumorMulticlassClassifier
            classifier = BrainTumorMulticlassClassifier()
            
            # Use preprocessed array directly
            predictions = classifier.model.predict(preprocessed, verbose=0)
            scores = predictions[0]
            predicted_idx = np.argmax(scores)
            predicted_class = classifier.classes[predicted_idx]
            confidence = float(scores[predicted_idx])
            
            result = {
                "success": True,
                "model_type": "brain_tumor",
                "prediction": {
                    "tumor_type": predicted_class,
                    "confidence": confidence,
                    "all_scores": dict(zip(classifier.classes, [float(s) for s in scores]))
                }
            }
            
            LOGGER.info(f"Brain Tumor Prediction: {predicted_class} (confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            LOGGER.error(f"Brain tumor prediction error: {e}")
            return {"success": False, "error": f"Brain tumor prediction failed: {str(e)}"}
    
    elif model_type == "alzheimer":
        # Preprocess for Alzheimer
        preprocessed = ImageValidator.preprocess_alzheimer_image(image_path)
        if preprocessed is None:
            return {"success": False, "error": "Failed to preprocess image for Alzheimer model"}
        
        # Import and use Alzheimer model
        try:
            from models.alzheimer_multiclass import AlzheimerMulticlassClassifier
            classifier = AlzheimerMulticlassClassifier()
            
            # Use preprocessed array directly
            predictions = classifier.model.predict(preprocessed, verbose=0)
            scores = predictions[0]
            predicted_idx = np.argmax(scores)
            predicted_class = classifier.class_names[predicted_idx]
            confidence = float(scores[predicted_idx])
            
            result = {
                "success": True,
                "model_type": "alzheimer",
                "prediction": {
                    "dementia_stage": predicted_class,
                    "confidence": confidence,
                    "all_scores": dict(zip(classifier.class_names, [float(s) for s in scores]))
                }
            }
            
            LOGGER.info(f"Alzheimer Prediction: {predicted_class} (confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            LOGGER.error(f"Alzheimer prediction error: {e}")
            return {"success": False, "error": f"Alzheimer prediction failed: {str(e)}"}
    
    else:
        return {"success": False, "error": f"Unknown model type: {model_type}"}
