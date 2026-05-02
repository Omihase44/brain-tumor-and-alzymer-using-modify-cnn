import argparse
import os
import sys
from collections import Counter

import tensorflow as tf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.tumor_model import build_tumor_grading_model
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


CLASS_NAMES = ["no tumor", "tumor"]
TUMOR_LABEL_MAP = {
    "no": "no tumor",
    "yes": "tumor",
}


def discover_tumor_samples(dataset_dir: str) -> list[Sample]:
    resolved_dir = os.path.abspath(dataset_dir)
    samples: list[Sample] = []
    for directory_entry in sorted(os.scandir(resolved_dir), key=lambda item: item.name.lower()):
        if not directory_entry.is_dir():
            continue
        normalized_label = TUMOR_LABEL_MAP.get(directory_entry.name.strip().lower())
        if normalized_label is None:
            continue
        for file_entry in sorted(os.scandir(directory_entry.path), key=lambda item: item.name.lower()):
            if file_entry.is_file():
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
    parser = argparse.ArgumentParser(description="Train the production tumor detector with stratified 80/20 split (train/validation).")
    parser.add_argument("--dataset-dir", default=os.path.join("dataset", "brain"))
    parser.add_argument("--output-dir", default=os.path.join("trained_models", "tumor"))
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

    samples = discover_tumor_samples(args.dataset_dir)
    if len(samples) < 20:
        raise ValueError("Not enough labeled tumor samples were found to train the detector.")

    train_samples, validation_samples = split_samples_train_validation(samples, seed=args.seed, train_ratio=0.80, validation_ratio=0.20)
    output_dir = ensure_directory(args.output_dir)
    image_size = (args.image_size, args.image_size)
    best_model_path = os.path.join(output_dir, "tumor_detector_best.h5")
    final_model_path = os.path.join(output_dir, "model.h5")
    metadata_path = os.path.join(output_dir, "tumor_detector_metadata.json")
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    split_manifest_path = os.path.join(output_dir, "dataset_split.csv")

    base_model = build_tumor_grading_model(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=len(CLASS_NAMES),
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

    class_weights = calculate_class_weights(train_samples, CLASS_NAMES)
    train_dataset = build_dataset(
        train_samples,
        CLASS_NAMES,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_dataset = build_dataset(
        validation_samples,
        CLASS_NAMES,
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
    train_metrics = evaluate_classifier(best_model, train_samples, CLASS_NAMES, image_size, args.batch_size, args.seed)
    validation_metrics = evaluate_classifier(best_model, validation_samples, CLASS_NAMES, image_size, args.batch_size, args.seed)
    best_model.save(final_model_path)

    print_evaluation_report("Training", train_samples, CLASS_NAMES, train_metrics)
    print_evaluation_report("Validation", validation_samples, CLASS_NAMES, validation_metrics)

    history_plots = save_history_plots(history.history, output_dir)
    confusion_artifacts = save_confusion_matrix_artifacts(validation_metrics.get("confusion_matrix", []), CLASS_NAMES, output_dir)
    split_manifest_path = save_split_manifest(
        split_manifest_path,
        _build_split_manifest_rows(train_samples, validation_samples),
    )
    write_json(metrics_path, validation_metrics)

    metadata = {
        "model_key": "brain_classifier",
        "task": "tumor_detection_binary",
        "class_names": CLASS_NAMES,
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
    write_model_accuracy_file("brain_classifier", validation_metrics)

    if args.activate:
        update_manifest_entry(
            "brain_classifier",
            os.path.abspath(final_model_path),
            os.path.abspath(metadata_path),
        )

    print("\n" + "="*70)
    print("TUMOR MODEL TRAINING SUMMARY")
    print("="*70)
    print(f"Train accuracy:      {train_metrics.get('accuracy'):.4f}")
    print(f"Validation accuracy: {validation_metrics.get('accuracy'):.4f}")
    print(f"Target accuracy:     {args.target_accuracy:.4f}")
    print(f"Target met:          {validation_metrics.get('accuracy', 0.0) >= args.target_accuracy}")
    print("="*70)


if __name__ == "__main__":
    main()
