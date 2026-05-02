from django.shortcuts import render, redirect
from django.contrib.sessions.models import Session
from django.contrib import messages
from django.conf import settings
import os
import logging
import numpy as np
import operator
from datetime import date, datetime
from .models import User_Details, Admin_Details

# Try to import the new Alzheimer classifier
try:
    from services.alzheimer_service import predict_alzheimer
    ALZHEIMER_MODEL_AVAILABLE = True
except ImportError:
    ALZHEIMER_MODEL_AVAILABLE = False
    logging.warning("Alzheimer model not available, falling back to legacy model")

# Fallback imports for legacy model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

LOGGER = logging.getLogger(__name__)




def home(request):
    return render(request, 'home.html', {})


def about(request):
    return render(request, "about.html", {})


def base(request):
    return render(request, "base.html", {})


def Admin_login(request):
    if request.method == 'POST':
        Username = request.POST['Username']
        password = request.POST['password']
        
        if Admin_Details.objects.filter(Username=Username, Password=password).exists():
                user = Admin_Details.objects.get(Username=Username, Password=password)
                request.session['type_id'] = 'Admin'
                request.session['username'] = Username
                request.session['login'] = 'Yes'
                return redirect('/')
        else:
            messages.info(request,'Invalid Credentials')
            return redirect('/Admin_login/')
    else:
        return render(request, 'Admin_login.html', {})


def analysis(request):
    """
    Unified analysis view for both Brain Tumor and Alzheimer detection
    """
    if request.method == 'POST':
        analysis_type = request.POST.get('analysis_type', 'brain')
        
        # Validate analysis type
        if analysis_type not in ['brain', 'alzheimer']:
            messages.error(request, 'Invalid analysis type specified.')
            return redirect('/')
        
        LOGGER.info(f"Analysis request received for type: {analysis_type}")
        
        name = request.POST.get('name')
        age = request.POST.get('age')
        contact = request.POST.get('contact')
        emailid = request.POST.get('email')
        username = request.POST.get('Username')
        image_file = request.FILES.get('Image1')

        # Check if username or email already exists
        if User_Details.objects.filter(Username=username).exists():
            messages.error(request, f'Username already taken for {analysis_type} analysis.')
            return redirect(f'/{analysis_type}/')
        elif User_Details.objects.filter(Email=emailid).exists():
            messages.error(request, f'Email already taken for {analysis_type} analysis.')
            return redirect(f'/{analysis_type}/')

        # Save image file to media folder
        if image_file:
            image_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
            with open(image_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            try:
                if analysis_type == 'brain':
                    # Brain Tumor Analysis
                    LOGGER.info(f"Starting Brain Tumor analysis for image: {image_path}")
                    if MULTICLASS_MODEL_AVAILABLE:
                        LOGGER.info("Using new multiclass brain tumor model for prediction")
                        result = predict_brain_tumor(image_path=image_path)

                        if result.get('success'):
                            prediction_data = result['prediction']
                            tumor_type = prediction_data.get('tumor_type', 'Unknown')
                            confidence = prediction_data.get('confidence', 0)
                            predicted_class = tumor_type.lower().replace(' tumor', '').replace(' ', '')
                            grade = prediction_data.get('grade', 'N/A')
                            model_accuracy = prediction_data.get('model_accuracy', {})

                            LOGGER.info(f"Brain Tumor Prediction: {tumor_type} (confidence: {confidence:.4f}), Grade: {grade}")
                        else:
                            messages.error(request, f"Brain tumor prediction failed: {result.get('error', 'Unknown error')}")
                            return redirect('/brain/')
                    else:
                        # Fallback to legacy model
                        LOGGER.info("Using legacy brain tumor model for prediction")
                        loaded_model = load_model('brain_model_new.h5', compile=False)
                        img = image.load_img(image_path, target_size=(224, 224))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        predictions = loaded_model.predict(x)[0]
                        class_names = ['glioma tumor', 'meningioma tumor', 'no tumor', 'pituitary tumor']
                        sorted_prediction = sorted(zip(class_names, predictions), key=operator.itemgetter(1), reverse=True)
                        predicted_class = sorted_prediction[0][0]
                        tumor_type = predicted_class
                        confidence = sorted_prediction[0][1]
                        grade = 'N/A'
                        model_accuracy = {}

                    # Determine symptoms and treatment
                    symptoms, treatment = get_symptoms_and_treatment(predicted_class)

                    # Save user details
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

                    messages.success(request, 'Brain tumor analysis completed successfully! Results available in the admin page.')

                    # Context for brain tumor results
                    context = {
                        'analysis_type': 'brain',
                        'predicted_class': predicted_class,
                        'tumor_type': tumor_type,
                        'confidence': f"{confidence*100:.2f}%" if isinstance(confidence, float) else confidence,
                        'grade': grade,
                        'symptoms': symptoms,
                        'treatment': treatment,
                        'model_accuracy': model_accuracy,
                        'model_validation_accuracy': f"{model_accuracy.get('validation_accuracy', 0)*100:.2f}%" if model_accuracy else "N/A",
                        'model_test_accuracy': f"{model_accuracy.get('test_accuracy', 0)*100:.2f}%" if model_accuracy else "N/A",
                    }

                elif analysis_type == 'alzheimer':
                    # Alzheimer Analysis
                    LOGGER.info(f"Starting Alzheimer analysis for image: {image_path}")
                    if ALZHEIMER_MODEL_AVAILABLE:
                        prediction_result = predict_alzheimer(image_path=image_path)

                        if prediction_result["success"]:
                            predicted_class = prediction_result["prediction"]
                            confidence = prediction_result["confidence"]
                            model_accuracy = prediction_result["model_accuracy"]
                            all_probabilities = prediction_result["all_probabilities"]
                            
                            LOGGER.info(f"Alzheimer analysis successful: {predicted_class} (confidence: {confidence:.4f})")
                        else:
                            messages.error(request, f'Alzheimer prediction failed: {prediction_result.get("error", "Unknown error")}')
                            return redirect('/Alzhimers/')
                    else:
                        # Fallback to legacy model
                        LOGGER.warning("Alzheimer model not available, using legacy model")
                        loaded_model = load_model('alz_model_new.h5', compile=False)
                        img = image.load_img(image_path, target_size=(224, 224))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        predictions = loaded_model.predict(x)[0]
                        class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
                        prediction = dict(zip(class_names, predictions))
                        sorted_prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                        predicted_class = sorted_prediction[0][0]
                        confidence = sorted_prediction[0][1]
                        model_accuracy = 0.0
                        all_probabilities = {}

                    # Determine symptoms and treatment
                    symptoms, treatment = get_symptoms_and_treatment(predicted_class)

                    # Save user details
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

                    messages.success(request, 'Alzheimer analysis completed successfully! Results available in the admin page.')

                    # Context for Alzheimer results
                    context = {
                        'analysis_type': 'alzheimer',
                        'predicted_class': predicted_class,
                        'confidence': f"{confidence*100:.2f}%" if isinstance(confidence, float) else confidence,
                        'model_accuracy': model_accuracy,
                        'symptoms': symptoms,
                        'treatment': treatment,
                        'all_probabilities': all_probabilities if 'all_probabilities' in locals() else {},
                        'model_validation_accuracy': f"{model_accuracy*100:.2f}%" if isinstance(model_accuracy, float) else "N/A",
                    }

                else:
                    messages.error(request, 'Invalid analysis type selected.')
                    return redirect('/analysis/')

                LOGGER.info(f"Analysis completed for {analysis_type}: {context}")
                return render(request, 'analysis.html', context)

            except Exception as e:
                LOGGER.error(f"Error during {analysis_type} analysis: {e}", exc_info=True)
                messages.error(request, f'An error occurred during {analysis_type} analysis: {str(e)}')
                return redirect(f'/{analysis_type}/')
        else:
            messages.error(request, 'No image uploaded.')
            return redirect('/analysis/')
    else:
        # GET request - show the unified analysis form
        return render(request, 'analysis.html', {'analysis_type': request.GET.get('type', 'brain')})







def get_symptoms_and_treatment(predicted_class):
    """
    Get symptoms and treatment based on predicted tumor class.
    Supports both new (glioma, meningioma, notumor, pituitary) and old class names.
    """
    predicted_class_lower = str(predicted_class).lower().replace(' tumor', '').replace(' ', '')
    
    # Brain tumor classes (new model)
    if predicted_class_lower in ['glioma', 'gliomat tumour']:
        symptoms = "Headache, Nausea or vomiting, Confusion or a decline in brain function, Memory loss, Personality changes or irritability."
        treatment = "Chemotherapy drugs can be taken in pill form (orally) or injected into a vein (intravenously). Surgery and radiation therapy may also be recommended."
    elif predicted_class_lower in ['meningioma', 'meningiomat tumour']:
        symptoms = "Hearing loss or ringing in the ears, Memory loss, Loss of smell, Seizures, Vision problems."
        treatment = "The first treatment for a meningioma is surgery if possible. The goal is to obtain tissue and remove as much tumor as possible without causing more symptoms. Radiation therapy may follow."
    elif predicted_class_lower in ['notumor', 'no tumor', 'notumort umor']:
        symptoms = "No problems detected. Brain scan appears normal."
        treatment = "No treatment required. Regular check-ups recommended."
    elif predicted_class_lower in ['pituitary', 'pituitary tumor', 'pituitary t umor']:
        symptoms = "Vision problems, Unexplained tiredness, Mood changes, Irritability, Unexplained changes in menstrual cycles, Erectile dysfunction."
        treatment = "Surgery, Radiation therapy, Medications to manage hormone levels, Replacement of pituitary hormones."
    # Legacy brain tumor names
    elif predicted_class == "glioma tumor":
        symptoms = "Headache, Nausea or vomiting, Confusion or a decline in brain function, Memory loss, Personality changes or irritability."
        treatment = "Chemotherapy drugs can be taken in pill form (orally) or injected into a vein (intravenously)."
    elif predicted_class == "meningioma tumor":
        symptoms = "Hearing loss or ringing in the ears, Memory loss, Loss of smell, Seizures."
        treatment = "The first treatment for a malignant meningioma is surgery, if possible. The goal of surgery is to obtain tissue to determine the tumor type and to remove as much tumor as possible without causing more symptoms for the person."
    elif predicted_class == "no tumor":
        symptoms = "No problems detected."
        treatment = "No problems detected."
    elif predicted_class == "pituitary tumor":
        symptoms = "Vision problems, Unexplained tiredness, Mood changes, Irritability, Unexplained changes in menstrual cycles, Erectile dysfunction."
        treatment = "Surgery, Radiation therapy, Medications, Replacement of pituitary hormones."
    # Alzheimer's classes (new model)
    elif predicted_class in ["Mild Dementia", "MildDementia"]:
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
        # Default values if the predicted class is unknown
        symptoms = "Symptoms not available."
        treatment = "Treatment not available."
    
    return symptoms, treatment



def brain(request):
    """Redirect to unified analysis view for brain tumor"""
    if request.method == 'POST':
        # Add analysis type to POST data
        request.POST = request.POST.copy()
        request.POST['analysis_type'] = 'brain'
        return analysis(request)
    else:
        return analysis(request)


def Alzhimers(request):
    """Redirect to unified analysis view for Alzheimer"""
    if request.method == 'POST':
        # Add analysis type to POST data
        request.POST = request.POST.copy()
        request.POST['analysis_type'] = 'alzheimer'
        return analysis(request)
    else:
        return analysis(request)





def logout(request):
    Session.objects.all().delete()
    return redirect('/')






def View_Users(request):
    if request.method == 'POST':
        return redirect('/View_Users/')
    else:
        sty = User_Details.objects.all()
        return render(request, 'View_Users.html', {'sty':sty})

