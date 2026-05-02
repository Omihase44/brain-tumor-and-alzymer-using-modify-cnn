#!/usr/bin/env python3
"""
Alzheimer Disease Classification CLI Tool
Command-line interface for Alzheimer disease stage analysis using the improved CNN model
"""

import argparse
import os
import sys
from pathlib import Path
import io

# Handle Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.alzheimer_service import AlzheimerPredictionService
from models.alzheimer_multiclass import AlzheimerMulticlassClassifier

def main():
    parser = argparse.ArgumentParser(
        description='Alzheimer Disease Classification CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python alzheimer_cli.py image.jpg
  python alzheimer_cli.py image.jpg --show-accuracy
  python alzheimer_cli.py /path/to/brain/scan.png --show-accuracy
        '''
    )
    
    parser.add_argument('image_path', help='Path to the brain MRI image file')
    parser.add_argument(
        '--show-accuracy', 
        action='store_true', 
        help='Show detailed model accuracy information'
    )
    
    args = parser.parse_args()
    
    image_path = args.image_path
    show_accuracy = args.show_accuracy
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}", file=sys.stderr)
        return 1
    
    print(f"🧠 Analyzing brain MRI image for Alzheimer disease: {image_path}")
    print("=" * 60)
    
    try:
        # Perform prediction
        result = AlzheimerPredictionService.predict_from_image_file(image_path=image_path)
        
        if result.get('success'):
            prediction = result.get('prediction', 'Unknown')
            confidence = result.get('confidence', 0)
            model_accuracy = result.get('model_accuracy', 0)
            all_probabilities = result.get('all_probabilities', {})
            
            # Display results
            print("✅ Analysis Complete")
            print()
            
            # Determine if Alzheimer disease is detected
            alzheimer_detected = prediction not in ['Non Demented', 'Unknown']
            stage = prediction
            
            if alzheimer_detected:
                print(f"⚠️  ALZHEIMER DISEASE DETECTED: {stage.upper()}")
            else:
                print("✅ NO ALZHEIMER DISEASE DETECTED")
            
            print(f"Confidence: {confidence*100:.2f}%")
            print(f"Stage: {stage}")
            print()
            
            # Show scores for all classes
            print("Stage Probabilities:")
            for class_name, score in all_probabilities.items():
                marker = '←' if score == confidence else '  '
                print(f"  {marker} {class_name}: {score*100:.2f}%")
            
            print()
            
            # Show model accuracy if requested
            if show_accuracy:
                classifier = AlzheimerMulticlassClassifier()
                model_info = classifier.get_model_info()
                
                print("🤖 AI Model Performance:")
                print(f"  Validation Accuracy: {model_info.get('final_val_accuracy', model_accuracy)*100:.2f}%")
                print(f"  Training Accuracy: {model_info.get('final_train_accuracy', 0)*100:.2f}%")
                print(f"  Test Accuracy: {model_info.get('accuracy', model_accuracy)*100:.2f}%")
                print(f"  Model Type: Alzheimer EfficientNetB0")
                print(f"  Classes: {len(all_probabilities)}")
                print(f"  Epochs Trained: {model_info.get('epochs_completed', 0)}")
                
                improvements = model_info.get('improvements', [])
                if improvements:
                    print("  Improvements:")
                    for improvement in improvements:
                        print(f"    • {improvement}")
                
                print()
            print("🏥 Clinical Insights:")
            if alzheimer_detected:
                if 'mild' in stage.lower():
                    print("  • Early stage Alzheimer disease")
                    print("  • Consider cognitive assessment and baseline evaluation")
                    print("  • Lifestyle interventions may help slow progression")
                elif 'moderate' in stage.lower():
                    print("  • Moderate Alzheimer disease progression")
                    print("  • May require assistance with daily activities")
                    print("  • Consider cholinesterase inhibitors")
                elif 'very mild' in stage.lower():
                    print("  • Very mild cognitive impairment")
                    print("  • Monitor for progression to Alzheimer disease")
                    print("  • Regular cognitive assessments recommended")
            else:
                print("  • Normal cognitive status indicated")
                print("  • Continue regular health screenings")
                print("  • Maintain brain-healthy lifestyle")
            
            print()
            print("⚠️  IMPORTANT: This is AI-assisted analysis only.")
            print("   Final diagnosis requires comprehensive clinical evaluation,")
            print("   including neuropsychological testing and physician review.")
            
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ Analysis failed: {error}", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
