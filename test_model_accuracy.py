from models.brain_tumor_multiclass import BrainTumorMulticlassClassifier
from models.alzheimer_multiclass import AlzheimerMulticlassClassifier
import numpy as np
from PIL import Image as PILImage

print('=== TESTING BOTH MODELS WITH REAL IMAGES ===')

# Test Brain Tumor Model
print('\n--- BRAIN TUMOR MODEL ---')
try:
    bt_classifier = BrainTumorMulticlassClassifier()
    glioma_img = PILImage.open('dataset/brain/Training/glioma/Tr-gl_1.jpg').convert('RGB')
    glioma_array = np.array(glioma_img)
    bt_result = bt_classifier.predict(glioma_array)

    print('Glioma Image Test:')
    print('  Detected:', bt_result.get('detected'))
    print('  Prediction:', bt_result.get('tumor_type'))
    print('  Confidence:', f"{bt_result.get('confidence'):.4f}")
    print('  Model Accuracy:', bt_result.get('model_accuracy'))

except Exception as e:
    print('Brain Tumor Model Error:', str(e))

# Test Alzheimer Model
print('\n--- ALZHEIMER MODEL ---')
try:
    alz_classifier = AlzheimerMulticlassClassifier()
    # Use file path instead of numpy array
    alz_result = alz_classifier.predict('dataset/alzheimer/non/train__No Impairment__NoImpairment (1).jpg')

    if alz_result:
        print('Non-Demented Image Test:')
        print('  Prediction:', alz_result.get('prediction'))
        print('  Confidence:', f"{alz_result.get('confidence'):.4f}")
        print('  Model Accuracy:', alz_result.get('model_accuracy'))
        print('  All Probabilities:', alz_result.get('all_probabilities'))
    else:
        print('Alzheimer Model returned None')

except Exception as e:
    print('Alzheimer Model Error:', str(e))