"""
Enhanced Brain Tumor Classification Views (4-Class CNN)
Updated views using the new multi-class CNN model
"""

import os
import logging
import numpy as np
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

try:
    from services.brain_tumor_service import predict_brain_tumor
    BRAIN_TUMOR_SERVICE_AVAILABLE = True
except ImportError:
    BRAIN_TUMOR_SERVICE_AVAILABLE = False
    logging.warning("Brain tumor service not available")

try:
    from services.alzheimer_service import predict_alzheimer
    ALZHEIMER_SERVICE_AVAILABLE = True
except ImportError:
    ALZHEIMER_SERVICE_AVAILABLE = False
    logging.warning("Alzheimer service not available")

from b_tumor.models import User_Details

LOGGER = logging.getLogger(__name__)

# Tumor type details
TUMOR_DETAILS = {
    "glioma": {
        "name": "Glioma Tumor",
        "grade": "Grade III",
        "symptoms": "Headache, Nausea or vomiting, Confusion or a decline in brain function, Memory loss, Personality changes or irritability.",
        "treatment": "Chemotherapy drugs can be taken in pill form (orally) or injected into a vein (intravenously). Surgery and radiation therapy may also be recommended."
    },
    "meningioma": {
        "name": "Meningioma Tumor",
        "grade": "Grade II",
        "symptoms": "Hearing loss or ringing in the ears, Memory loss, Loss of smell, Seizures, Vision problems.",
        "treatment": "The first treatment for a meningioma is surgery if possible. The goal is to obtain tissue and remove as much tumor as possible without causing more symptoms. Radiation therapy may follow."
    },
    "notumor": {
        "name": "No Tumor",
        "grade": "None",
        "symptoms": "No problems detected. Brain scan appears normal.",
        "treatment": "No treatment required. Regular check-ups recommended."
    },
    "pituitary": {
        "name": "Pituitary Tumor",
        "grade": "Grade IV",
        "symptoms": "Vision problems, Unexplained tiredness, Mood changes, Irritability, Unexplained changes in menstrual cycles, Erectile dysfunction.",
        "treatment": "Surgery, Radiation therapy, Medications to manage hormone levels, Replacement of pituitary hormones."
    }
}


def brain_tumor_classification_v2(request):
    """
    Enhanced brain tumor classification view using 4-class CNN
    """
    if request.method == 'POST':
        try:
            # Get form data
            name = request.POST.get('name', '').strip()
            age = request.POST.get('age', '').strip()
            contact = request.POST.get('contact', '').strip()
            emailid = request.POST.get('email', '').strip()
            username = request.POST.get('Username', '').strip()
            image_file = request.FILES.get('Image1')
            
            # Validate inputs
            if not all([name, age, contact, emailid, username, image_file]):
                messages.error(request, 'All fields are required.')
                return redirect('/brain/')
            
            # Check if username or email already exists
            if User_Details.objects.filter(Username=username).exists():
                messages.error(request, 'Username already taken.')
                return redirect('/brain/')
            elif User_Details.objects.filter(Email=emailid).exists():
                messages.error(request, 'Email already taken.')
                return redirect('/brain/')
            
            # Save image file
            image_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
            with open(image_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)
            
            # Make prediction - try both models to determine image type
            if not BRAIN_TUMOR_SERVICE_AVAILABLE:
                messages.error(request, 'Classification service unavailable. Please try again later.')
                return redirect('/brain/')
            
            # Save image temporarily for processing
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                for chunk in image_file.chunks():
                    temp_file.write(chunk)
                temp_image_path = temp_file.name
            
            try:
                # Get predictions from both models
                brain_result = predict_brain_tumor(image_path=temp_image_path)
                
                alzheimer_result = None
                if ALZHEIMER_SERVICE_AVAILABLE:
                    alzheimer_result = predict_alzheimer(image_path=temp_image_path)
                
                # Determine which model is more appropriate
                analysis_type = 'brain'  # default
                final_result = brain_result
                
                if alzheimer_result and alzheimer_result.get('success'):
                    # Compare confidence scores to determine image type
                    brain_confidence = brain_result.get('prediction', {}).get('confidence', 0)
                    alz_confidence = alzheimer_result.get('confidence', 0)
                    
                    # If Alzheimer model has higher confidence, use it
                    if alz_confidence > brain_confidence:
                        analysis_type = 'alzheimer'
                        final_result = alzheimer_result
                        logging.info(f"Image classified as Alzheimer (confidence: {alz_confidence:.4f}) over Brain Tumor (confidence: {brain_confidence:.4f})")
                    else:
                        logging.info(f"Image classified as Brain Tumor (confidence: {brain_confidence:.4f}) over Alzheimer (confidence: {alz_confidence:.4f})")
                else:
                    logging.info("Alzheimer service not available, using brain tumor classification only")
                
                if not final_result.get('success'):
                    error_msg = final_result.get('error', 'Unknown error')
                    messages.error(request, f"Classification failed: {error_msg}")
                    return redirect('/brain/')
                
                # Process the result based on analysis type
                if analysis_type == 'brain':
                    prediction = final_result['prediction']
                    tumor_class = prediction.get('tumor_class', 'unknown')
                    tumor_type = prediction.get('tumor_type', 'Unknown')
                    confidence = prediction.get('confidence', 0)
                    
                    # Get tumor details
                    details_key = 'notumor' if tumor_type == 'No Tumor' else tumor_class
                    tumor_info = TUMOR_DETAILS.get(details_key, TUMOR_DETAILS['notumor'])
                    symptoms = tumor_info['symptoms']
                    treatment = tumor_info['treatment']
                    grade = tumor_info['grade']
                    
                    LOGGER.info(f"Brain Tumor Prediction: {tumor_class} (confidence: {confidence:.4f})")
                    
                    # Save to database
                    user = User_Details(
                        Name=name,
                        Contact=contact,
                        Age=age,
                        Email=emailid,
                        Username=username,
                        Image1=image_file,
                        Symptoms=symptoms,
                        Treatment=treatment,
                        Class_detected=tumor_type
                    )
                    user.save()
                    
                    messages.success(request, 'Brain tumor analysis completed successfully!')
                    
                    # Return results
                    context = {
                        'predicted_class': tumor_class,
                        'predicted_type': tumor_type,
                        'grade': grade,
                        'symptoms': symptoms,
                        'treatment': treatment,
                        'confidence': f"{confidence*100:.2f}%",
                        'all_scores': prediction.get('scores', {}),
                        'analysis_type': 'brain'
                    }
                    
                    return render(request, 'analysis.html', context)
                    
                else:  # Alzheimer analysis
                    predicted_class = final_result.get('prediction', 'Unknown')
                    confidence = final_result.get('confidence', 0)
                    model_accuracy = final_result.get('model_accuracy', 0)
                    
                    # Get Alzheimer symptoms and treatment
                    if predicted_class in ["Mild Dementia", "MildDementia"]:
                        symptoms = "Forgetfulness, Trouble with problem-solving, Difficulty completing familiar tasks, Confusion with time or place, Mood changes."
                        treatment = "Medications (cholinesterase inhibitors), Cognitive therapy, Lifestyle changes, Regular exercise, Healthy diet."
                    elif predicted_class in ["Moderate Dementia", "ModerateDementia"]:
                        symptoms = "Worsening memory, Difficulty recognizing family and friends, Increased confusion, Difficulty speaking, Anxiety or aggression, Hallucinations."
                        treatment = "Medications, Supportive therapies, Supervision and assistance, Caregiver support, Safety measures at home."
                    elif predicted_class in ["Non Demented", "NonDementia"]:
                        symptoms = "No significant cognitive impairment observed. Normal cognitive function."
                        treatment = "No specific treatment required. Regular health check-ups recommended."
                    elif predicted_class in ["Very mild Dementia", "VeryMildDementia"]:
                        symptoms = "Subtle cognitive decline, May not be noticeable to others, Slight memory lapses, Difficulty with complex tasks."
                        treatment = "Lifestyle changes, Regular monitoring, Cognitive training, Social engagement, Healthy diet and exercise."
                    else:
                        symptoms = "Symptoms analysis in progress."
                        treatment = "Consultation with healthcare provider recommended."
                    
                    LOGGER.info(f"Alzheimer Prediction: {predicted_class} (confidence: {confidence:.4f})")
                    
                    # Save to database
                    user = User_Details(
                        Name=name,
                        Contact=contact,
                        Age=age,
                        Email=emailid,
                        Username=username,
                        Image1=image_file,
                        Symptoms=symptoms,
                        Treatment=treatment,
                        Class_detected=predicted_class
                    )
                    user.save()
                    
                    messages.success(request, 'Alzheimer analysis completed successfully!')
                    
                    # Return results
                    context = {
                        'predicted_class': predicted_class,
                        'confidence': f"{confidence*100:.2f}%",
                        'model_accuracy': f"{model_accuracy*100:.2f}%" if model_accuracy else "N/A",
                        'symptoms': symptoms,
                        'treatment': treatment,
                        'analysis_type': 'alzheimer'
                    }
                    
                    return render(request, 'analysis.html', context)
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
            
        except Exception as e:
            LOGGER.error(f"Error in brain tumor classification: {e}")
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect('/brain/')
    
    else:
        return render(request, 'brain.html', {})


def brain_tumor_info(request):
    """
    Provide information about brain tumors and the classification system
    """
    context = {
        'tumor_types': TUMOR_DETAILS,
        'num_classes': len(TUMOR_DETAILS)
    }
    return render(request, 'brain_info.html', context)


def brain_tumor_api_classify(request):
    """
    API endpoint for brain tumor classification (for AJAX requests)
    Expects multipart form data with 'image' field
    """
    import json
    from django.http import JsonResponse
    
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        
        # Save temporarily
        temp_path = os.path.join(settings.MEDIA_ROOT, f'temp_{image_file.name}')
        with open(temp_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)
        
        # Predict
        if not BRAIN_TUMOR_SERVICE_AVAILABLE:
            return JsonResponse({'error': 'Service unavailable'}, status=503)
        
        prediction_result = predict_brain_tumor(image_path=temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if prediction_result.get('success'):
            return JsonResponse({
                'success': True,
                'prediction': prediction_result['prediction']
            })
        else:
            return JsonResponse({
                'error': prediction_result.get('error', 'Prediction failed')
            }, status=400)
        
    except Exception as e:
        LOGGER.error(f"Error in API classification: {e}")
        return JsonResponse({'error': str(e)}, status=500)
