import argparse
import os
import sys
from collections import Counter
from typing import Optional

import tensorflow as tf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.alzheimer_model import (
    ALZHEIMER_STAGE_CLASSES,
    build_alzheimer_staging_model,
    normalize_alzheimer_stage_label,
)
from training.pipeline import (
    Sample,
    build_callbacks,
    build_dataset,
    calculate_class_weights,
    compile_classifier,
    ensure_directory,
    evaluate_classifier,
    print_evaluation_report,
    save_confusion_matrix_artifacts,
    save_history_plots,
    save_split_manifest,
    split_samples_train_validation,
    update_manifest_entry,
    with_augmentation,
    write_json,
    write_model_accuracy_file,
)


FOLDER_STAGE_FALLBACK = {
    "non": "NonDemented",
    "mild": "Very Mild",
    "moderate": "Mild",
    "severe": "Moderate",
}


def _infer_stage_from_filename(filename: str) -> Optional[str]:
    normalized = filename.lower().replace("_", " ").replace("-", " ")
    if "non dementia" in normalized or "nondementia" in normalized or "non demented" in normalized:
        return "NonDemented"
    if "very mild" in normalized:
        return "Very Mild"
    if "moderate impairment" in normalized or "moderate dementia" in normalized:
        return "Moderate"
    if "mild impairment" in normalized or "mild dementia" in normalized:
        return "Mild"
    return None


def discover_alzheimer_samples(dataset_dir: str) -> list[Sample]:
    resolved_dir = os.path.abspath(dataset_dir)
    samples: list[Sample] = []
    for directory_entry in sorted(os.scandir(resolved_dir), key=lambda item: item.name.lower()):
        if not directory_entry.is_dir():
            continue
        for file_entry in sorted(os.scandir(directory_entry.path), key=lambda item: item.name.lower()):
            if not file_entry.is_file():
                continue
            inferred_label = _infer_stage_from_filename(file_entry.name)
            if inferred_label is None:
                fallback_label = FOLDER_STAGE_FALLBACK.get(directory_entry.name.lower())
                inferred_label = normalize_alzheimer_stage_label(fallback_label)
            normalized_label = normalize_alzheimer_stage_label(inferred_label)
            if normalized_label in ALZHEIMER_STAGE_CLASSES:
                samples.append(Sample(path=file_entry.path, label=normalized_label))
    return samples


def _build_split_manifest_rows(train_samples, validation_samples):
    rows = []
    for split_name, split_samples in (
        ("train", train_samples),
        ("validation", validation_samples),
    ):
        for sample in split_samples:
            rows.append(
                {
                    "split": split_name,
                    "label": sample.label,
                    "path": sample.path,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the upgraded Alzheimer staging classifier with a stratified 80/20 split (train/validation).")
    parser.add_argument("--dataset-dir", default=os.path.join("dataset", "alzheimer"))
    parser.add_argument("--output-dir", default=os.path.join("trained_models", "alz"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--backbone", choices=("cnn", "mobilenetv2", "resnet50", "vgg16"), default="cnn")
    parser.add_argument("--fine-tune-layers", type=int, default=16)
    parser.add_argument("--weights", default="none")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.35)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    samples = discover_alzheimer_samples(args.dataset_dir)
    if len(samples) < 20:
        raise ValueError("Not enough labeled Alzheimer samples were found to train the classifier.")

    train_samples, validation_samples = split_samples_train_validation(samples, seed=args.seed, train_ratio=0.80, validation_ratio=0.20)
    output_dir = ensure_directory(args.output_dir)
    image_size = (args.image_size, args.image_size)
    best_model_path = os.path.join(output_dir, "alz_classifier_best.h5")
    final_model_path = os.path.join(output_dir, "model.h5")
    metadata_path = os.path.join(output_dir, "alz_classifier_metadata.json")
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    split_manifest_path = os.path.join(output_dir, "dataset_split.csv")

    base_model = build_alzheimer_staging_model(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=len(ALZHEIMER_STAGE_CLASSES),
        variant=args.backbone,
        weights=args.weights,
        dropout_rate=args.dropout_rate,
        fine_tune_layers=args.fine_tune_layers,
    )
    if base_model is None:
        raise RuntimeError("TensorFlow Keras is unavailable in the current environment.")
    model = compile_classifier(
        with_augmentation(base_model, (args.image_size, args.image_size, 3)),
        learning_rate=args.learning_rate,
    )

    class_weights = calculate_class_weights(train_samples, ALZHEIMER_STAGE_CLASSES)
    train_dataset = build_dataset(
        train_samples,
        ALZHEIMER_STAGE_CLASSES,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_dataset = build_dataset(
        validation_samples,
        ALZHEIMER_STAGE_CLASSES,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=build_callbacks(best_model_path),
        class_weight=class_weights,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(best_model_path, compile=False)
    train_metrics = evaluate_classifier(best_model, train_samples, ALZHEIMER_STAGE_CLASSES, image_size, args.batch_size, args.seed)
    validation_metrics = evaluate_classifier(best_model, validation_samples, ALZHEIMER_STAGE_CLASSES, image_size, args.batch_size, args.seed)
    best_model.save(final_model_path)

    print_evaluation_report("Training", train_samples, ALZHEIMER_STAGE_CLASSES, train_metrics)
    print_evaluation_report("Validation", validation_samples, ALZHEIMER_STAGE_CLASSES, validation_metrics)

    history_plots = save_history_plots(history.history, output_dir)
    confusion_artifacts = save_confusion_matrix_artifacts(validation_metrics.get("confusion_matrix", []), ALZHEIMER_STAGE_CLASSES, output_dir)
    split_manifest_path = save_split_manifest(
        split_manifest_path,
        _build_split_manifest_rows(train_samples, validation_samples),
    )
    write_json(metrics_path, validation_metrics)

    metadata = {
        "model_key": "alzheimer_classifier",
        "task": "alzheimer_staging_multiclass",
        "class_names": ALZHEIMER_STAGE_CLASSES,
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "image_size": list(image_size),
        "backbone": args.backbone,
        "weights": args.weights,
        "learning_rate": args.learning_rate,
        "dropout_rate": args.dropout_rate,
        "target_accuracy": args.target_accuracy,
        "target_met": bool(validation_metrics.get("accuracy", 0.0) >= args.target_accuracy),
        "sample_count": len(samples),
        "class_distribution": dict(Counter(sample.label for sample in samples)),
        "class_weights": class_weights,
        "train_count": len(train_samples),
        "validation_count": len(validation_samples),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "history": history.history,
        "history_plots": history_plots,
        "confusion_artifacts": confusion_artifacts,
        "split_manifest_path": os.path.abspath(split_manifest_path),
        "best_model_path": os.path.abspath(best_model_path),
        "final_model_path": os.path.abspath(final_model_path),
        "metrics_path": os.path.abspath(metrics_path),
    }
    write_json(metadata_path, metadata)
    write_model_accuracy_file("alzheimer_classifier", validation_metrics)

    if args.activate:
        update_manifest_entry(
            "alzheimer_classifier",
            os.path.abspath(final_model_path),
            os.path.abspath(metadata_path),
        )

    print("\n" + "="*70)
    print("ALZHEIMER MODEL TRAINING SUMMARY")
    print("="*70)
    print(f"Train accuracy:      {train_metrics.get('accuracy'):.4f}")
    print(f"Validation accuracy: {validation_metrics.get('accuracy'):.4f}")
    print(f"Target accuracy:     {args.target_accuracy:.4f}")
    print(f"Target met:          {validation_metrics.get('accuracy', 0.0) >= args.target_accuracy}")
    print("="*70)


if __name__ == "__main__":
    main()
