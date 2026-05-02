"""
Brain Tumor Classification Routes (4-Class CNN)
Endpoints for Glioma, Meningioma, No Tumor, Pituitary classification
"""

import json
import logging
from typing import Dict
import numpy as np
from flask import Blueprint, request, jsonify

from models.brain_tumor_multiclass import get_multiclass_classifier
from utils.image_processing import preprocess_classifier_image

LOGGER = logging.getLogger(__name__)

# Create blueprint
brain_tumor_multiclass_bp = Blueprint(
    'brain_tumor_multiclass',
    __name__,
    url_prefix='/api/brain-tumor'
)


@brain_tumor_multiclass_bp.route('/classify', methods=['POST'])
def classify_tumor():
    """
    Classify brain tumor from MRI image.
    
    Expected form data:
    - image: Image file (jpg, png, etc.)
    
    Returns:
    - JSON with classification results
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Read and preprocess image
        from PIL import Image
        import io
        
        img = Image.open(io.BytesIO(file.read()))
        image_array = np.array(img)
        
        # Get classifier and predict
        classifier = get_multiclass_classifier()
        result = classifier.predict(image_array)
        
        return jsonify({
            'success': True,
            'prediction': result
        }), 200
        
    except Exception as e:
        LOGGER.error(f"Error in tumor classification: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@brain_tumor_multiclass_bp.route('/info', methods=['GET'])
def get_classifier_info():
    """Get information about the classifier model."""
    try:
        classifier = get_multiclass_classifier()
        info = classifier.get_model_info()
        
        return jsonify({
            'success': True,
            'classifier_info': info
        }), 200
        
    except Exception as e:
        LOGGER.error(f"Error getting classifier info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@brain_tumor_multiclass_bp.route('/classes', methods=['GET'])
def get_classes():
    """Get list of tumor classes."""
    try:
        classifier = get_multiclass_classifier()
        
        return jsonify({
            'success': True,
            'classes': classifier.classes,
            'class_labels': classifier.class_labels,
            'grade_mapping': classifier.grade_mapping
        }), 200
        
    except Exception as e:
        LOGGER.error(f"Error getting classes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def register_multiclass_routes(app):
    """Register the multiclass tumor classification routes with Flask app."""
    app.register_blueprint(brain_tumor_multiclass_bp)
