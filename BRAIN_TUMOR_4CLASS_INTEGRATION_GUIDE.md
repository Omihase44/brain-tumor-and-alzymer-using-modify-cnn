# Brain Tumor 4-Class CNN Classification - Integration Guide

## Overview

This document describes the implementation of a **4-class Brain Tumor Classification CNN model** that classifies brain MRI images into:
- **Glioma Tumor** (Grade III)
- **Meningioma Tumor** (Grade II)  
- **No Tumor** (None)
- **Pituitary Tumor** (Grade IV)

The model achieves approximately **99% accuracy** on the test dataset.

## Model Architecture

### CNN Structure
```
Input: 150x150x3 RGB Images
↓
Conv2D(32, 4x4) + MaxPool(3x3)
↓
Conv2D(64, 4x4) + MaxPool(3x3)
↓
Conv2D(128, 4x4) + MaxPool(3x3)
↓
Conv2D(128, 4x4)
↓
Flatten
↓
Dense(512, relu) + Dropout(0.5)
↓
Dense(4, softmax)
Output: 4-class probabilities
```

### Training Configuration
- **Image Size**: 150x150 pixels
- **Batch Size**: 32
- **Epochs**: 40
- **Optimizer**: Adam (lr=0.001, β₁=0.869, β₂=0.995)
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: Rotation, brightness, shear, horizontal flip
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

## Project Structure

```
e:\New folder (2)\
├── training/
│   ├── train_tumor_multiclass.py          # Training script
│   ├── train_tumor_model.py               # Existing binary classifier
│   └── pipeline.py                        # Training utilities
├── models/
│   ├── brain_tumor_multiclass.py          # 4-class classifier module
│   ├── tumor_model.py                     # Existing tumor model
│   └── alzheimer_model.py
├── services/
│   ├── brain_tumor_service.py             # Django integration service
│   ├── classification.py                  # Main classification service
│   └── ...
├── routes/
│   ├── brain_tumor_multiclass_routes.py   # Flask/API routes
│   └── ...
├── b_tumor/
│   ├── views_brain_tumor_v2.py            # Enhanced Django views
│   ├── views.py                           # Existing views
│   └── models.py
├── dataset/
│   └── brain/
│       ├── Training/
│       │   ├── glioma/          (1400 images)
│       │   ├── meningioma/      (1400 images)
│       │   ├── notumor/         (1400 images)
│       │   └── pituitary/       (1400 images)
│       └── Testing/
│           ├── glioma/          (400 images)
│           ├── meningioma/      (400 images)
│           ├── notumor/         (400 images)
│           └── pituitary/       (400 images)
└── trained_models/
    ├── brain_tumor_cnn_multiclass.h5             # Trained model
    └── tumor_model_multiclass_metadata.json      # Model metadata
```

## Files Created/Modified

### New Files
1. **training/train_tumor_multiclass.py** - Training script for the 4-class model
2. **models/brain_tumor_multiclass.py** - Classifier module for inference
3. **services/brain_tumor_service.py** - Django service wrapper
4. **routes/brain_tumor_multiclass_routes.py** - API endpoints
5. **b_tumor/views_brain_tumor_v2.py** - Enhanced Django views

### Key Components

#### 1. Training Script (`training/train_tumor_multiclass.py`)
```bash
# Run training
python training/train_tumor_multiclass.py
```

Features:
- Loads dataset from `dataset/brain/Training` and `dataset/brain/Testing`
- Builds the 4-class CNN model
- Trains with data augmentation
- Saves model to `trained_models/brain_tumor_cnn_multiclass.h5`
- Saves metadata to `trained_models/tumor_model_multiclass_metadata.json`

#### 2. Classifier Module (`models/brain_tumor_multiclass.py`)
```python
from models.brain_tumor_multiclass import get_multiclass_classifier

classifier = get_multiclass_classifier()
result = classifier.predict(image_array)
```

Returns:
```json
{
    "detected": true,
    "classification": "Glioma Tumor",
    "tumor_type": "Glioma Tumor",
    "tumor_class": "glioma",
    "grade": "Grade III",
    "confidence": 0.9847,
    "scores": {
        "glioma": 0.9847,
        "meningioma": 0.0089,
        "notumor": 0.0043,
        "pituitary": 0.0021
    },
    "backend": "keras_multiclass"
}
```

#### 3. Django Service (`services/brain_tumor_service.py`)
```python
from services.brain_tumor_service import predict_brain_tumor

# From file path
result = predict_brain_tumor(image_path="/path/to/image.jpg")

# From bytes
result = predict_brain_tumor(image_bytes=image_data)

# From numpy array
result = predict_brain_tumor(image_array=np.array([...]))
```

#### 4. Enhanced Django Views (`b_tumor/views_brain_tumor_v2.py`)
```python
# Register the view in urls.py:
# path('brain/', brain_tumor_classification_v2, name='brain_tumor_v2'),
```

## Integration Steps

### Step 1: Train the Model
```bash
cd "e:\New folder (2)"
.\.venv\Scripts\python.exe training\train_tumor_multiclass.py
```

Expected output:
- Model accuracy > 95% on test set
- Model saved to `trained_models/brain_tumor_cnn_multiclass.h5`

### Step 2: Verify Model Exists
```bash
# Check if model file exists
ls trained_models/brain_tumor_cnn_multiclass.h5
```

### Step 3: Update Django URLs (if using new views)
Add to `b_tumor/urls.py`:
```python
from b_tumor.views_brain_tumor_v2 import (
    brain_tumor_classification_v2,
    brain_tumor_api_classify,
    brain_tumor_info
)

urlpatterns = [
    # ... existing patterns ...
    path('brain/v2/', brain_tumor_classification_v2, name='brain_tumor_v2'),
    path('api/brain/classify/', brain_tumor_api_classify, name='brain_tumor_api'),
    path('brain/info/', brain_tumor_info, name='brain_tumor_info'),
]
```

### Step 4: Test the Integration
```python
from services.brain_tumor_service import predict_brain_tumor
import numpy as np
from PIL import Image

# Test with a real image
img = Image.open('dataset/brain/Testing/glioma/image.jpg')
image_array = np.array(img)

result = predict_brain_tumor(image_array=image_array)
print(result)
```

## Usage Examples

### Example 1: Direct Classifier Usage
```python
from models.brain_tumor_multiclass import get_multiclass_classifier
import numpy as np

classifier = get_multiclass_classifier()

# Predict on numpy array
image = np.random.rand(150, 150, 3) * 255  # Random image
result = classifier.predict(image)

print(f"Tumor Type: {result['tumor_type']}")
print(f"Grade: {result['grade']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Example 2: Django View Usage
```python
# In Django view
from services.brain_tumor_service import predict_brain_tumor

if request.FILES.get('image'):
    image_file = request.FILES['image']
    prediction = predict_brain_tumor(image_bytes=image_file.read())
    
    if prediction['success']:
        tumor_type = prediction['prediction']['tumor_type']
        confidence = prediction['prediction']['confidence']
        # Pass to template
```

### Example 3: API Endpoint
```bash
# POST request to classify image
curl -X POST http://localhost:8000/api/brain/classify/ \
  -F "image=@path/to/brain_mri.jpg"

# Response:
{
    "success": true,
    "prediction": {
        "detected": true,
        "tumor_type": "Glioma Tumor",
        "grade": "Grade III",
        "confidence": 0.9847
    }
}
```

## Model Classes and Grade Information

| Class | Label | Grade | Typical Symptoms | Treatment |
|-------|-------|-------|------------------|-----------|
| glioma | Glioma Tumor | Grade III | Headache, nausea, memory loss | Chemotherapy, surgery |
| meningioma | Meningioma Tumor | Grade II | Hearing loss, seizures | Surgery, radiation |
| pituitary | Pituitary Tumor | Grade IV | Vision problems, hormonal changes | Surgery, medications |
| notumor | No Tumor | None | None | No treatment |

## Performance Metrics

Based on the notebook analysis:
- **Test Accuracy**: ~99%
- **Training Accuracy**: >98%
- **Total Parameters**: 495,972 (1.89 MB)
- **Input Size**: 150×150×3 RGB
- **Output**: 4-class probabilities

### Per-Class Metrics
- **Glioma**: Precision ~98%, Recall ~99%
- **Meningioma**: Precision ~99%, Recall ~99%
- **Pituitary**: Precision ~99%, Recall ~98%
- **No Tumor**: Precision ~99%, Recall ~99%

## Troubleshooting

### Issue: Model Not Found
```
Error: Model not found at trained_models/brain_tumor_cnn_multiclass.h5
```
**Solution**: Run the training script first
```bash
python training/train_tumor_multiclass.py
```

### Issue: TensorFlow Version Mismatch
```
Error: Model was saved with ... but tf.keras version is different
```
**Solution**: Ensure TensorFlow version matches the training environment
```bash
pip install tensorflow==2.x.x  # Use same version as training
```

### Issue: Image Preprocessing Error
```
Error: Could not convert image
```
**Solution**: Ensure image is in valid format (JPG, PNG)
```python
from PIL import Image
img = Image.open('image.jpg')
# Convert to RGB if grayscale
if img.mode != 'RGB':
    img = img.convert('RGB')
```

## Performance Optimization

### For Faster Inference
1. Use GPU if available
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. Use batch predictions for multiple images
   ```python
   results = classifier.predict_batch(image_batch)
   ```

### For Deployment
1. Quantize the model for smaller size
2. Use TensorFlow Lite for mobile deployment
3. Set up model caching

## Dataset Information

**Total Images**: 7,200
- Training: 5,600 (1,400 per class)
- Testing: 1,600 (400 per class)

**Dataset Source**: 
- Compiled from figshare and Br35H datasets
- Available on Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

**Image Properties**:
- Format: JPEG
- Varying original sizes (normalized to 150×150 during training)
- Grayscale and RGB mixed (handled in preprocessing)

## Future Enhancements

1. **Model Improvements**
   - Fine-tuning with transfer learning (ResNet, VGG)
   - Attention mechanisms
   - Ensemble methods

2. **Feature Enhancements**
   - Tumor localization (segmentation)
   - Grade prediction
   - Confidence calibration

3. **Deployment**
   - ONNX export for cross-platform compatibility
   - TensorFlow Lite for mobile
   - Docker containerization

## References

- **Original Notebook**: `cnn-brain-tumor-classification-99-accuracy.ipynb`
- **Kaggle Dataset**: Brain Tumor MRI Dataset
- **TensorFlow Documentation**: https://www.tensorflow.org/
- **Keras Documentation**: https://keras.io/

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the notebook for model architecture details
3. Verify dataset is properly structured
4. Ensure all dependencies are installed

