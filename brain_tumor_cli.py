#!/usr/bin/env python3
"""
Brain Tumor Classification CLI Tool
Command-line interface for brain tumor analysis using the improved CNN model
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

from services.brain_tumor_service import predict_brain_tumor
from models.brain_tumor_multiclass import get_multiclass_classifier

def main():
    parser = argparse.ArgumentParser(
        description='Brain Tumor Classification CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python brain_tumor_cli.py image.jpg
  python brain_tumor_cli.py image.jpg --show-accuracy
  python brain_tumor_cli.py /path/to/mri/scan.png --show-accuracy
        '''
    )
    
    parser.add_argument('image_path', help='Path to the MRI image file')
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
    
    print(f"🧠 Analyzing brain MRI image: {image_path}")
    print("=" * 60)
    
    try:
        # Perform prediction
        result = predict_brain_tumor(image_path=image_path)
        
        if result.get('success'):
            prediction = result.get('prediction', {})
            
            # Display results
            print("✅ Analysis Complete")
            print()
            
            tumor_detected = prediction.get('detected', False)
            tumor_type = prediction.get('tumor_type', 'Unknown')
            confidence = prediction.get('confidence', 0)
            grade = prediction.get('grade', 'Unknown')
            
            if tumor_detected:
                print(f"⚠️  TUMOR DETECTED: {tumor_type.upper()}")
            else:
                print("✅ NO TUMOR DETECTED")
            
            print(f"Confidence: {confidence*100:.2f}%")
            print(f"Grade: {grade}")
            print()
            
            # Show scores for all classes
            scores = prediction.get('scores', {})
            print("Class Probabilities:")
            for class_name, score in scores.items():
                marker = '←' if score == confidence else '  '
                print(f"  {marker} {class_name}: {score*100:.2f}%")
            
            print()
            
            # Show model accuracy if requested
            if show_accuracy:
                classifier = get_multiclass_classifier()
                model_info = classifier.get_model_info()
                accuracy_info = model_info.get('accuracy', {})
                
                print("🤖 AI Model Performance:")
                print(f"  Validation Accuracy: {accuracy_info.get('validation_accuracy', 0)*100:.2f}%")
                print(f"  Test Accuracy: {accuracy_info.get('test_accuracy', 0)*100:.2f}%")
                print(f"  Training Accuracy: {accuracy_info.get('training_accuracy', 0)*100:.2f}%")
                print(f"  Epochs Trained: {accuracy_info.get('epochs_trained', 0)}")
                print(f"  Best Epoch: {accuracy_info.get('best_epoch', 0)}")
                
                improvements = accuracy_info.get('improvements', [])
                if improvements:
                    print("  Improvements:")
                    for improvement in improvements:
                        print(f"    • {improvement}")
                
                print()
            
            # Show clinical insights
            print("🏥 Clinical Insights:")
            if tumor_detected:
                if 'glioma' in tumor_type.lower():
                    print("  • Requires immediate neurosurgical consultation")
                    print("  • Consider MRI with contrast for further evaluation")
                elif 'meningioma' in tumor_type.lower():
                    print("  • Often benign, but location-dependent treatment")
                    print("  • Regular monitoring may be sufficient")
                elif 'pituitary' in tumor_type.lower():
                    print("  • May affect hormone production")
                    print("  • Endocrinological evaluation recommended")
            else:
                print("  • Normal brain MRI findings")
                print("  • Clinical correlation still recommended")
            
            print()
            print("⚠️  IMPORTANT: This is AI-assisted analysis only.")
            print("   Final diagnosis requires physician review.")
            
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
