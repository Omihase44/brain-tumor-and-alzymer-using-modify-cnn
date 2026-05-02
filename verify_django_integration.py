"""
Integration Verification Script
Verifies that the new brain tumor model is integrated into Django views
"""

import os
import sys
from pathlib import Path

# Add repo to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

def check_model_file():
    """Check if trained model exists"""
    model_path = REPO_ROOT / "trained_models" / "brain_tumor_cnn_multiclass.h5"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model file exists: {model_path}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"⏳ Model file not yet created: {model_path}")
        return False


def check_views_integration():
    """Check if views.py has been updated"""
    views_path = REPO_ROOT / "b_tumor" / "views.py"
    if views_path.exists():
        content = views_path.read_text()
        if "brain_tumor_service" in content and "predict_brain_tumor" in content:
            print("✓ Views.py integrated with new model")
            return True
        elif "MULTICLASS_MODEL_AVAILABLE" in content:
            print("✓ Views.py has new model support code")
            return True
        else:
            print("⚠ Views.py may not have new model integration")
            return False
    return False


def check_service_available():
    """Check if brain tumor service is available"""
    try:
        from services.brain_tumor_service import predict_brain_tumor
        print("✓ Brain tumor service module available")
        return True
    except ImportError as e:
        print(f"⚠ Brain tumor service not fully available: {e}")
        return False


def check_classifier_available():
    """Check if classifier module is available"""
    try:
        from models.brain_tumor_multiclass import get_multiclass_classifier
        print("✓ Multiclass classifier module available")
        return True
    except ImportError as e:
        print(f"⚠ Multiclass classifier not available: {e}")
        return False


def main():
    print("=" * 70)
    print("DJANGO INTEGRATION VERIFICATION")
    print("=" * 70)
    print()
    
    checks = [
        ("Model File", check_model_file),
        ("Views Integration", check_views_integration),
        ("Service Module", check_service_available),
        ("Classifier Module", check_classifier_available),
    ]
    
    results = {}
    for check_name, check_func in checks:
        print(f"\n[{check_name}]")
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"✗ Error: {e}")
            results[check_name] = False
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓" if result else "⏳/⚠"
        print(f"{status} {check_name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if not check_model_file():
        print("\n⏳ Waiting for model training to complete...")
        print("   Run this script again after training finishes (Epoch 40/40)")
    else:
        if passed == total:
            print("\n✓ DJANGO INTEGRATION COMPLETE!")
            print("  You can now use the new model in your Django project")
        else:
            print("\n⚠ Some checks failed, please review above")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
