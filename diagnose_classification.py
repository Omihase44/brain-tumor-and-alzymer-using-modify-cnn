"""
Brain Tumor Classification Pipeline Diagnostic Script
=====================================================
Validates that the classification pipeline is working correctly by:
1. Loading the model and printing its architecture summary
2. Running predictions on test images from each class
3. Printing full softmax probabilities for every prediction
4. Computing a confusion matrix on the test set
5. Reporting whether Keras model or heuristic fallback is in use
"""

import os
import sys
import logging
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Set up logging so we see all diagnostic messages
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("diagnose")

# Ensure project root is on the path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_model_and_inspect():
    """Load the tumor model and print diagnostic information."""
    from models.tumor_model import TumorGradingModel, TUMOR_TYPE_CLASSES, TUMOR_DISPLAY_LABELS

    print("\n" + "=" * 70)
    print("STEP 1: MODEL LOADING DIAGNOSTIC")
    print("=" * 70)

    model_instance = TumorGradingModel()

    print(f"  Model path:       {model_instance.model_path}")
    print(f"  Model file exists: {os.path.exists(model_instance.model_path)}")
    print(f"  Backend:          {model_instance.backend}")
    print(f"  Class names:      {model_instance.class_names}")
    print(f"  Task:             {model_instance.task}")
    print(f"  Strict loading:   {model_instance.strict_loading}")
    print(f"  Default classes:  {TUMOR_TYPE_CLASSES}")
    print(f"  Display labels:   {TUMOR_DISPLAY_LABELS}")

    if model_instance.model is not None:
        print(f"\n  ✅ Keras model loaded successfully!")
        print(f"  Input shape:  {model_instance.model.input_shape}")
        print(f"  Output shape: {model_instance.model.output_shape}")
        output_units = model_instance.model.output_shape[-1]
        print(f"  Output units: {output_units}")
        if output_units != len(model_instance.class_names):
            print(f"  ⚠️  WARNING: Output units ({output_units}) != class count ({len(model_instance.class_names)})")
            print(f"     This means the wrong model file is loaded!")
        else:
            print(f"  ✅ Output units match class count ({output_units})")
    else:
        print(f"\n  ❌ Keras model is None — all predictions will use HEURISTIC fallback!")
        print(f"     Checked path: {model_instance.model_path}")

    return model_instance


def test_single_prediction(model_instance, image_path, expected_class):
    """Run a single prediction and return results."""
    from utils.image_processing import preprocess_classifier_image_rgb

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None, f"Failed to read image: {image_path}"

    # Use the model's actual input shape to determine target size
    if model_instance.model is not None:
        input_shape = model_instance.model.input_shape
        if input_shape and len(input_shape) >= 3:
            target_size = (int(input_shape[1]), int(input_shape[2]))
        else:
            target_size = (150, 150)
    else:
        target_size = (150, 150)

    # Use the RGB preprocessor (matches inference pipeline)
    prepared = preprocess_classifier_image_rgb(image, target_size=target_size)
    result = model_instance.predict(image, prepared_image=prepared)

    return result, None


def run_test_set_evaluation(model_instance, test_dir):
    """Run evaluation on the test dataset and compute confusion matrix."""
    print("\n" + "=" * 70)
    print("STEP 2: TEST SET EVALUATION")
    print("=" * 70)

    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"  ❌ Test directory not found: {test_dir}")
        return

    class_folders = sorted([d for d in test_path.iterdir() if d.is_dir()])
    print(f"  Found class folders: {[f.name for f in class_folders]}")

    # Internal class names (matching training)
    internal_classes = list(model_instance.class_names)
    num_classes = len(internal_classes)

    # Display label → internal label mapping
    from models.tumor_model import TUMOR_DISPLAY_LABELS
    display_to_internal = {v: k for k, v in TUMOR_DISPLAY_LABELS.items()}

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    total = 0
    correct = 0
    per_class_results = {cls: {"correct": 0, "total": 0, "predictions": Counter()} for cls in internal_classes}

    max_per_class = 50  # Limit for speed

    for folder in class_folders:
        folder_name = folder.name.lower()
        if folder_name not in internal_classes:
            print(f"  ⚠️  Skipping folder '{folder.name}' — not in class list")
            continue

        true_idx = internal_classes.index(folder_name)
        image_files = sorted(folder.glob("*"))[:max_per_class]
        print(f"\n  Testing class '{folder_name}' ({len(image_files)} images)...")

        for img_path in image_files:
            if not img_path.is_file():
                continue

            result, error = test_single_prediction(model_instance, str(img_path), folder_name)
            if error:
                continue

            # Map display label back to internal label
            predicted_display = result.get("tumor_type", "")
            predicted_internal = display_to_internal.get(predicted_display, predicted_display)
            if predicted_internal.lower() in internal_classes:
                pred_idx = internal_classes.index(predicted_internal.lower())
            else:
                pred_idx = -1

            total += 1
            per_class_results[folder_name]["total"] += 1
            per_class_results[folder_name]["predictions"][predicted_internal] += 1

            if pred_idx == true_idx:
                correct += 1
                per_class_results[folder_name]["correct"] += 1

            if pred_idx >= 0:
                confusion[true_idx][pred_idx] += 1

    # Print results
    print("\n" + "=" * 70)
    print("STEP 3: RESULTS SUMMARY")
    print("=" * 70)

    accuracy = correct / total if total > 0 else 0
    print(f"\n  Overall Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"  Backend used:    {model_instance.backend}")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':>15}", end="")
    for cls in internal_classes:
        print(f"  {cls:>12}", end="")
    print()
    for i, cls in enumerate(internal_classes):
        print(f"  {cls:>15}", end="")
        for j in range(num_classes):
            print(f"  {confusion[i][j]:>12}", end="")
        print()

    print(f"\n  Per-Class Results:")
    for cls in internal_classes:
        info = per_class_results[cls]
        cls_acc = info["correct"] / info["total"] if info["total"] > 0 else 0
        print(f"    {cls:>15}: {info['correct']}/{info['total']} = {cls_acc:.2%}")
        print(f"                   Predicted as: {dict(info['predictions'])}")

    # Diagnosis
    print("\n" + "=" * 70)
    print("STEP 4: DIAGNOSIS")
    print("=" * 70)

    if model_instance.backend == "heuristic":
        print("  ❌ PROBLEM: Using HEURISTIC backend — Keras model is NOT being used")
        print("     Check that the model file exists and has the correct output shape")
    elif accuracy < 0.5:
        print("  ❌ PROBLEM: Accuracy is very low — likely preprocessing mismatch")
        print("     Check BGR/RGB order and normalization")
    elif accuracy < 0.8:
        print("  ⚠️  WARNING: Accuracy is below expected — possible issues")
    else:
        print("  ✅ Classification appears to be working correctly")

    return confusion


def run_quick_spot_check(model_instance, test_dir):
    """Run 1 image from each class and show full details."""
    print("\n" + "=" * 70)
    print("STEP 5: SPOT CHECK (1 image per class)")
    print("=" * 70)

    test_path = Path(test_dir)
    if not test_path.exists():
        return

    for folder in sorted(test_path.iterdir()):
        if not folder.is_dir():
            continue

        images = list(folder.glob("*"))[:1]
        if not images:
            continue

        img_path = images[0]
        result, error = test_single_prediction(model_instance, str(img_path), folder.name)

        print(f"\n  True class:    {folder.name}")
        print(f"  Image:         {img_path.name}")
        if error:
            print(f"  Error:         {error}")
        else:
            print(f"  Predicted:     {result.get('tumor_type')}")
            print(f"  Confidence:    {result.get('confidence')}")
            print(f"  Backend:       {result.get('backend')}")
            print(f"  Detected:      {result.get('detected')}")
            print(f"  Grade:         {result.get('grade')}")
            match = folder.name.lower() in str(result.get("tumor_type", "")).lower() or \
                    (folder.name.lower() == "notumor" and "No Tumor" in str(result.get("tumor_type", "")))
            print(f"  Correct:       {'✅' if match else '❌'}")


def main():
    print("\n" + "=" * 70)
    print("BRAIN TUMOR CLASSIFICATION PIPELINE DIAGNOSTIC")
    print("=" * 70)

    # Step 1: Load and inspect model
    model_instance = load_model_and_inspect()

    # Step 2-4: Test set evaluation
    test_dir = os.path.join(REPO_ROOT, "dataset", "brain", "Testing")
    if os.path.exists(test_dir):
        run_test_set_evaluation(model_instance, test_dir)
        run_quick_spot_check(model_instance, test_dir)
    else:
        print(f"\n  ⚠️  Test dataset not found at: {test_dir}")
        print("  Skipping evaluation. Place test images in dataset/brain/Testing/")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
