import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from services.enhancement import ImageEnhancementService
from services.model_metrics import write_model_accuracy_registry_entry
from utils.tensorflow_compat import DEFAULT_MODEL_MANIFEST_PATH

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - only exercised when TensorFlow is available
    raise RuntimeError("TensorFlow is required to run the training scripts.") from exc


@dataclass(frozen=True)
class Sample:
    path: str
    label: str


_ENHANCEMENT_SERVICE = ImageEnhancementService()
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def ensure_directory(path: str) -> str:
    resolved_path = os.path.abspath(path)
    os.makedirs(resolved_path, exist_ok=True)
    return resolved_path


def relative_to_repo(path: str) -> str:
    resolved_path = os.path.abspath(path)
    try:
        return os.path.relpath(resolved_path, _REPO_ROOT).replace("\\", "/")
    except ValueError:
        return resolved_path


def _to_json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: str, payload: dict) -> str:
    ensure_directory(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(_to_json_compatible(payload), file_handle, indent=2)
    return os.path.abspath(path)


def _enhance_training_image(image: np.ndarray) -> np.ndarray:
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(bgr_image, None, 4, 4, 7, 15)
    enhanced = _ENHANCEMENT_SERVICE.enhance(denoised)["enhanced_image"]
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)


def load_rgb_image(path: str, image_size: tuple[int, int], apply_enhancement: bool = True) -> np.ndarray:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        image_array = np.asarray(rgb_image, dtype=np.uint8)
        if apply_enhancement:
            image_array = _enhance_training_image(image_array)
        resized = cv2.resize(image_array, image_size, interpolation=cv2.INTER_CUBIC)
        return np.clip(resized.astype(np.float32) / 255.0, 0.0, 1.0)


def build_dataset(
    samples: Sequence[Sample],
    class_names: Sequence[str],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    apply_enhancement: bool = True,
):
    class_index = {label: index for index, label in enumerate(class_names)}
    num_classes = len(class_names)

    def generator():
        for sample in samples:
            image = load_rgb_image(sample.path, image_size, apply_enhancement=apply_enhancement)
            target = np.zeros(num_classes, dtype=np.float32)
            target[class_index[sample.label]] = 1.0
            yield image, target

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(max(len(samples), 1), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_samples_three_way(
    samples: Sequence[Sample],
    seed: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if not samples:
        raise ValueError("At least one sample is required to build a dataset split.")
    if not math.isclose(train_ratio + validation_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    labels = [sample.label for sample in samples]
    stratify_labels = labels if len(set(labels)) > 1 else None
    train_samples, temp_samples = train_test_split(
        list(samples),
        test_size=(validation_ratio + test_ratio),
        random_state=seed,
        stratify=stratify_labels,
    )
    temp_labels = [sample.label for sample in temp_samples]
    validation_portion = validation_ratio / (validation_ratio + test_ratio)
    validation_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1.0 - validation_portion),
        random_state=seed,
        stratify=temp_labels if len(set(temp_labels)) > 1 else None,
    )
    return train_samples, validation_samples, test_samples


def split_samples_train_validation(
    samples: Sequence[Sample],
    seed: int,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.20,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into training and validation sets with stratification (80:20 by default)."""
    if not samples:
        raise ValueError("At least one sample is required to build a dataset split.")
    if not math.isclose(train_ratio + validation_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("Train and validation ratios must sum to 1.0.")

    labels = [sample.label for sample in samples]
    stratify_labels = labels if len(set(labels)) > 1 else None
    train_samples, validation_samples = train_test_split(
        list(samples),
        test_size=validation_ratio,
        random_state=seed,
        stratify=stratify_labels,
        shuffle=True,
    )
    return train_samples, validation_samples


def calculate_class_weights(samples: Sequence[Sample], class_names: Sequence[str]) -> Dict[int, float]:
    counts = Counter(sample.label for sample in samples)
    if not counts:
        return {}

    total_samples = float(sum(counts.values()))
    class_count = float(len(class_names))
    weights = {}
    for index, class_name in enumerate(class_names):
        label_count = float(counts.get(class_name, 0))
        if label_count <= 0:
            continue
        weights[index] = total_samples / (class_count * label_count)
    return weights


def with_augmentation(model, input_shape: tuple[int, int, int]):
    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.12)(x)
    x = layers.RandomTranslation(0.04, 0.04)(x)
    x = layers.RandomContrast(0.12)(x)
    random_brightness_layer = getattr(layers, "RandomBrightness", None)
    if random_brightness_layer is not None:
        x = random_brightness_layer(0.10)(x)
    outputs = model(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{model.name}_augmented")


def compile_classifier(model, learning_rate: float = 1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_callbacks(best_model_path: str, patience: int = 5):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), verbose=1),
        tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, mode="max"),
    ]


def _candidate_scores(model, dataset, class_names: Sequence[str]) -> np.ndarray:
    raw_scores = np.asarray(model.predict(dataset, verbose=0))
    if raw_scores.ndim == 1:
        raw_scores = np.expand_dims(raw_scores, axis=0)
    return raw_scores[:, : len(class_names)]


def evaluate_classifier(
    model,
    samples: Sequence[Sample],
    class_names: Sequence[str],
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> dict:
    if not samples:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": None,
            "sample_count": 0,
            "class_names": list(class_names),
            "confusion_matrix": [],
            "per_class": {},
        }

    dataset = build_dataset(
        samples,
        class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    scores = _candidate_scores(model, dataset, class_names)
    y_true = np.asarray([class_names.index(sample.label) for sample in samples], dtype=np.int64)
    y_pred = np.argmax(scores, axis=1)

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class = {}
    per_class_precision, per_class_recall, per_class_f1, supports = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    for class_index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": round(float(per_class_precision[class_index]), 4),
            "recall": round(float(per_class_recall[class_index]), 4),
            "f1_score": round(float(per_class_f1[class_index]), 4),
            "support": int(supports[class_index]),
        }

    roc_auc = None
    try:
        if len(class_names) == 2 and len(np.unique(y_true)) > 1:
            positive_index = 1
            roc_auc = float(roc_auc_score((y_true == positive_index).astype(np.int32), scores[:, positive_index]))
        elif len(class_names) > 2 and len(np.unique(y_true)) > 1:
            y_true_one_hot = np.eye(len(class_names))[y_true]
            roc_auc = float(
                roc_auc_score(
                    y_true_one_hot,
                    scores,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except Exception:
        roc_auc = None

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "sample_count": len(samples),
        "class_names": list(class_names),
        "confusion_matrix": confusion.astype(int).tolist(),
        "per_class": per_class,
    }


def print_evaluation_report(
    eval_name: str,
    samples: Sequence[Sample],
    class_names: Sequence[str],
    metrics: dict,
) -> None:
    """Print evaluation metrics and confusion matrix."""
    print(f"\n{'='*70}")
    print(f"{eval_name.upper()} SET EVALUATION")
    print(f"{'='*70}")
    print(f"Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
    print(f"Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0.0):.4f}")
    print(f"F1 Score:  {metrics.get('f1_score', 0.0):.4f}")
    if metrics.get('roc_auc') is not None:
        print(f"ROC AUC:   {metrics.get('roc_auc'):.4f}")
    
    print(f"\nConfusion Matrix:")
    confusion_data = np.array(metrics.get("confusion_matrix", []))
    if confusion_data.size > 0:
        print(confusion_data)
    
    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in metrics.get("per_class", {}).items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics.get('precision', 0.0):.4f}")
        print(f"    Recall:    {class_metrics.get('recall', 0.0):.4f}")
        print(f"    F1-Score:  {class_metrics.get('f1_score', 0.0):.4f}")
        print(f"    Support:   {class_metrics.get('support', 0)}")
    
    print(f"{'='*70}\n")


def save_split_manifest(path: str, split_rows: Iterable[dict]) -> str:
    ensure_directory(os.path.dirname(path))
    rows = list(split_rows)
    with open(path, "w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["split", "label", "path"])
        writer.writeheader()
        writer.writerows(rows)
    return os.path.abspath(path)


def _draw_line_chart(title: str, y_label: str, series_map: Dict[str, Sequence[float]], output_path: str) -> str:
    width, height = 960, 560
    margin_left, margin_right = 80, 30
    margin_top, margin_bottom = 60, 70
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin_left, margin_top, width - margin_right, height - margin_bottom), outline="#D4E2E8", width=2)
    draw.text((margin_left, 20), title, fill="#083A4F")
    draw.text((20, 20), y_label, fill="#637885")

    combined_values = [value for series in series_map.values() for value in series]
    if not combined_values:
        combined_values = [0.0, 1.0]
    y_min = min(combined_values)
    y_max = max(combined_values)
    if math.isclose(y_min, y_max):
        y_min = min(0.0, y_min)
        y_max = max(1.0, y_max)

    palette = ["#0E5A7A", "#3C8D5D", "#CC7A00", "#A94442"]
    for series_index, (label, series) in enumerate(series_map.items()):
        if not series:
            continue
        points = []
        for point_index, value in enumerate(series):
            x = margin_left + (chart_width * point_index / max(len(series) - 1, 1))
            y = margin_top + chart_height - ((float(value) - y_min) / max(y_max - y_min, 1e-6) * chart_height)
            points.append((x, y))
        draw.line(points, fill=palette[series_index % len(palette)], width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=palette[series_index % len(palette)])
        legend_x = margin_left + 10
        legend_y = height - margin_bottom + 18 + (series_index * 18)
        draw.line((legend_x, legend_y + 6, legend_x + 18, legend_y + 6), fill=palette[series_index % len(palette)], width=3)
        draw.text((legend_x + 24, legend_y), label, fill="#1F2F38")

    ensure_directory(os.path.dirname(output_path))
    image.save(output_path)
    return os.path.abspath(output_path)


def save_history_plots(history: dict, output_dir: str) -> Dict[str, str]:
    resolved_output_dir = ensure_directory(output_dir)
    accuracy_plot = _draw_line_chart(
        "Training Accuracy",
        "Accuracy",
        {
            "train_accuracy": history.get("accuracy", []),
            "val_accuracy": history.get("val_accuracy", []),
        },
        os.path.join(resolved_output_dir, "accuracy_history.png"),
    )
    loss_plot = _draw_line_chart(
        "Training Loss",
        "Loss",
        {
            "train_loss": history.get("loss", []),
            "val_loss": history.get("val_loss", []),
        },
        os.path.join(resolved_output_dir, "loss_history.png"),
    )
    return {
        "accuracy_history": accuracy_plot,
        "loss_history": loss_plot,
    }


def save_confusion_matrix_artifacts(confusion: Sequence[Sequence[int]], class_names: Sequence[str], output_dir: str) -> Dict[str, str]:
    resolved_output_dir = ensure_directory(output_dir)
    json_path = write_json(
        os.path.join(resolved_output_dir, "confusion_matrix.json"),
        {
            "class_names": list(class_names),
            "matrix": confusion,
        },
    )

    cell_size = 110
    header_size = 180
    dimension = header_size + (cell_size * len(class_names))
    image = Image.new("RGB", (dimension, dimension), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Confusion Matrix", fill="#083A4F")

    for index, label in enumerate(class_names):
        x = header_size + (index * cell_size)
        y = header_size + (index * cell_size)
        draw.text((x + 10, 85), str(label), fill="#083A4F")
        draw.text((20, y + 40), str(label), fill="#083A4F")

    max_value = max((value for row in confusion for value in row), default=1)
    for row_index, row in enumerate(confusion):
        for column_index, value in enumerate(row):
            x0 = header_size + (column_index * cell_size)
            y0 = header_size + (row_index * cell_size)
            intensity = int(235 - (180 * (float(value) / max(max_value, 1))))
            fill_color = (230, intensity, intensity)
            draw.rectangle((x0, y0, x0 + cell_size, y0 + cell_size), outline="#D4E2E8", fill=fill_color)
            draw.text((x0 + 42, y0 + 42), str(value), fill="#1F2F38")

    image_path = os.path.join(resolved_output_dir, "confusion_matrix.png")
    image.save(image_path)
    return {
        "confusion_matrix_json": json_path,
        "confusion_matrix_png": os.path.abspath(image_path),
    }


def update_manifest_entry(manifest_key: str, model_path: str, metadata_path: str, framework: str = "keras") -> str:
    manifest_path = os.environ.get("MODEL_MANIFEST_PATH", "").strip() or DEFAULT_MODEL_MANIFEST_PATH
    resolved_manifest_path = os.path.abspath(manifest_path)
    ensure_directory(os.path.dirname(resolved_manifest_path))

    payload = {}
    if os.path.exists(resolved_manifest_path):
        with open(resolved_manifest_path, "r", encoding="utf-8") as file_handle:
            try:
                payload = json.load(file_handle)
            except json.JSONDecodeError:
                payload = {}
    if not isinstance(payload, dict):
        payload = {}

    payload[manifest_key] = {
        "path": relative_to_repo(model_path),
        "metadata_path": relative_to_repo(metadata_path),
        "framework": framework,
    }
    with open(resolved_manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return resolved_manifest_path


def write_model_accuracy_file(model_key: str, metrics: dict) -> str:
    return write_model_accuracy_registry_entry(model_key, metrics)
