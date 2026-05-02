import importlib
import logging
from typing import Any, Dict, Optional

import numpy as np

from utils.image_processing import extract_radiology_features, preprocess_classifier_image, preprocess_classifier_image_rgb
from utils.tensorflow_compat import (
    ModelUnavailableError,
    get_tensorflow_keras,
    require_strict_model_loading,
    resolve_model_metadata,
    resolve_model_path,
    safe_load_keras_model,
)

LOGGER = logging.getLogger(__name__)

# Class labels must match the EXACT folder names used during training.
# ImageDataGenerator.flow_from_directory() sorts alphabetically:
#   glioma → 0, meningioma → 1, notumor → 2, pituitary → 3
TUMOR_TYPE_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# User-facing display labels (internal label → display label)
TUMOR_DISPLAY_LABELS = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

WHO_GRADE_CLASSES = ["Grade I", "Grade II", "Grade III", "Grade IV"]
# Keys must match the internal class names (without 'tumor' suffix)
TUMOR_STAGE_BY_TYPE = {
    "meningioma": "Grade II",
    "glioma": "Grade III",
    "pituitary": "Grade IV",
}


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def build_tumor_grading_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    variant: str = "cnn",
    weights: str = "imagenet",
    dense_units: int = 256,
    dropout_rate: float = 0.5,
    fine_tune_layers: int = 0,
) -> Optional[Any]:
    """Build a modified CNN or transfer-learning classifier for tumor staging."""
    tensorflow_keras = get_tensorflow_keras()
    if not tensorflow_keras["available"]:
        return None

    layers = tensorflow_keras["layers"]
    models = tensorflow_keras["models"]
    BatchNormalization = layers.BatchNormalization
    Conv2D = layers.Conv2D
    Dense = layers.Dense
    Dropout = layers.Dropout
    Flatten = layers.Flatten
    Input = layers.Input
    MaxPooling2D = layers.MaxPooling2D
    Model = models.Model

    inputs = Input(shape=input_shape)
    normalized_variant = str(variant or "cnn").strip().lower()

    if normalized_variant == "cnn":
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
    else:
        applications = importlib.import_module("tensorflow.keras.applications")
        preprocessing_layers = {
            "mobilenetv2": importlib.import_module("tensorflow.keras.applications.mobilenet_v2"),
            "resnet50": importlib.import_module("tensorflow.keras.applications.resnet50"),
            "vgg16": importlib.import_module("tensorflow.keras.applications.vgg16"),
        }
        if normalized_variant == "mobilenetv2":
            backbone_class = applications.MobileNetV2
            preprocessing = preprocessing_layers["mobilenetv2"].preprocess_input
        elif normalized_variant == "resnet50":
            backbone_class = applications.ResNet50
            preprocessing = preprocessing_layers["resnet50"].preprocess_input
        elif normalized_variant == "vgg16":
            backbone_class = applications.VGG16
            preprocessing = preprocessing_layers["vgg16"].preprocess_input
        else:
            raise ValueError(f"Unsupported tumor model variant: {variant}")

        rescaled_inputs = layers.Rescaling(255.0, name=f"{normalized_variant}_rescale")(inputs)
        processed_inputs = layers.Lambda(preprocessing, name=f"{normalized_variant}_preprocess")(rescaled_inputs)
        base_model = backbone_class(
            include_top=False,
            weights=None if str(weights).lower() == "none" else weights,
            input_shape=input_shape,
        )
        if fine_tune_layers > 0:
            base_model.trainable = True
            frozen_boundary = max(len(base_model.layers) - int(fine_tune_layers), 0)
            for layer in base_model.layers[:frozen_boundary]:
                layer.trainable = False
            for layer in base_model.layers[frozen_boundary:]:
                layer.trainable = True
        else:
            base_model.trainable = False
        x = base_model(processed_inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)

    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax", name="tumor_grade")(x)
    return Model(inputs=inputs, outputs=outputs, name=f"tumor_grading_model_{normalized_variant}")


class TumorGradingModel:
    """Tumor classifier that supports real models when available and heuristics otherwise."""

    def __init__(self, model_path: str = "trained_models/brain_tumor_cnn_multiclass.h5"):
        self.model_path = resolve_model_path(
            model_path,
            "BRAIN_MODEL_PATH",
            manifest_key="brain_classifier",
        )
        self.metadata = resolve_model_metadata("brain_classifier")
        self.class_names = self._resolve_class_names()
        self.task = str(self.metadata.get("task") or "multiclass").strip().lower()
        self.model = self._load_model()
        self.backend = "keras" if self.model is not None else "heuristic"
        self.strict_loading = require_strict_model_loading()
        LOGGER.info(
            "TumorGradingModel initialised: backend=%s, path=%s, classes=%s",
            self.backend, self.model_path, self.class_names,
        )

    def _load_model(self):
        return safe_load_keras_model(self.model_path)

    def _resolve_class_names(self):
        metadata_classes = self.metadata.get("class_names") or self.metadata.get("classes")
        if isinstance(metadata_classes, list) and metadata_classes:
            return [str(name) for name in metadata_classes]
        return list(TUMOR_TYPE_CLASSES)

    def predict(self, image: np.ndarray, prepared_image: Optional[np.ndarray] = None) -> Dict[str, object]:
        features = extract_radiology_features(image)
        model_output = self._predict_with_model(image, prepared_image=prepared_image)

        if model_output and self._is_binary_detection_task():
            tumor_detected = bool(model_output.get("detected"))
            detection_confidence = clamp(model_output.get("confidence", 0.0))
            if tumor_detected:
                tumor_type, type_confidence = self._heuristic_tumor_type(features)
            else:
                tumor_type, type_confidence = "notumor", detection_confidence
            backend = f"{self.backend}_binary_detection"
        elif model_output:
            tumor_type = model_output["classification"]
            type_confidence = model_output["confidence"]
            tumor_detected = tumor_type != "notumor"
            detection_confidence = type_confidence
            backend = self.backend
            LOGGER.info(
                "Keras model prediction: class=%s, confidence=%.4f",
                tumor_type, type_confidence,
            )
        elif self.strict_loading:
            raise ModelUnavailableError("Tumor classification model is unavailable.")
        else:
            LOGGER.warning(
                "Keras tumor model returned no output — falling back to HEURISTIC classifier. "
                "This means predictions are NOT from the trained CNN. "
                "Check model file: %s", self.model_path,
            )
            tumor_type, type_confidence = self._heuristic_tumor_type(features)
            tumor_detected = tumor_type != "notumor"
            detection_confidence = type_confidence
            backend = "heuristic"

        tumor_grade, grade_confidence = self._estimate_grade(features, tumor_type, tumor_detected)
        confidence = detection_confidence if not tumor_detected else max(detection_confidence, grade_confidence)

        # Map internal label to user-facing display label
        display_type = TUMOR_DISPLAY_LABELS.get(tumor_type, tumor_type)

        return {
            "detected": tumor_detected,
            "classification": display_type,
            "tumor_type": display_type,
            "grade": tumor_grade,
            "tumor_stage": tumor_grade,
            "confidence": round(confidence, 2),
            "type_confidence": round(type_confidence, 2),
            "stage_confidence": round(grade_confidence, 2),
            "backend": backend,
        }

    def _is_binary_detection_task(self) -> bool:
        return self.task in {"binary", "binary_detection", "tumor_detection_binary"} or len(self.class_names) == 2

    def _label_indicates_tumor(self, label: str) -> bool:
        normalized = str(label or "").strip().lower()
        return normalized in {"yes", "tumor", "tumor detected", "brain tumor", "positive"}

    def _predict_with_model(
        self,
        image: np.ndarray,
        prepared_image: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, object]]:
        if self.model is None:
            LOGGER.warning("Tumor model is None — cannot run Keras prediction.")
            return None

        try:
            if prepared_image is None:
                # Determine target size from the model's own input shape
                model_input_shape = self.model.input_shape  # e.g. (None, 150, 150, 3)
                if model_input_shape and len(model_input_shape) >= 3:
                    h, w = int(model_input_shape[1]), int(model_input_shape[2])
                    target_size = (h, w)
                else:
                    target_size = (150, 150)
                prepared_image = preprocess_classifier_image_rgb(image, target_size=target_size)
            prediction = self.model.predict(prepared_image, verbose=0)
            scores = np.asarray(prediction).squeeze()
            if scores.ndim != 1 or scores.shape[0] < len(self.class_names):
                LOGGER.warning(
                    "Model output shape mismatch: got %s scores but expected %d classes. "
                    "This likely means the wrong model file is loaded.",
                    scores.shape, len(self.class_names),
                )
                return None

            # Log full softmax probabilities for every class
            prob_summary = ", ".join(
                f"{self.class_names[i]}={scores[i]:.4f}"
                for i in range(len(self.class_names))
            )
            LOGGER.info("Tumor model softmax: [%s]", prob_summary)

            class_index = int(np.argmax(scores[: len(self.class_names)]))
            predicted_label = str(self.class_names[class_index])
            confidence = clamp(scores[class_index])
            if self._is_binary_detection_task():
                return {
                    "classification": predicted_label,
                    "detected": self._label_indicates_tumor(predicted_label),
                    "confidence": confidence,
                }
            return {
                "classification": predicted_label,
                "confidence": confidence,
            }
        except Exception as exc:
            LOGGER.exception("Keras tumor prediction failed: %s", exc)
            return None

    def _heuristic_tumor_type(self, features: Dict[str, float]) -> tuple[str, float]:
        tumor_score = clamp(
            (features["bright_ratio"] * 1.8)
            + (features["edge_density"] * 2.2)
            + (features["texture_score"] * 2.0)
            + (features["asymmetry"] * 3.5)
        )

        if tumor_score < 0.22:
            return "notumor", round(clamp(0.82 - tumor_score * 0.8, 0.1, 0.93), 2)

        if features["center_focus"] > 1.12 and features["bright_ratio"] > 0.08:
            tumor_type = "pituitary"
        elif features["asymmetry"] > 0.05 and features["edge_density"] > 0.04:
            tumor_type = "glioma"
        elif features["bright_ratio"] > 0.10:
            tumor_type = "meningioma"
        else:
            tumor_type = "glioma"

        return tumor_type, round(clamp(0.55 + tumor_score * 0.35, 0.55, 0.97), 2)

    def _estimate_grade(
        self,
        features: Dict[str, float],
        tumor_type: str,
        tumor_detected: bool,
    ) -> tuple[Optional[str], float]:
        if not tumor_detected:
            return None, 0.1

        mapped_grade = TUMOR_STAGE_BY_TYPE.get(str(tumor_type).strip().lower().replace(" tumor", ""))
        if mapped_grade:
            confidence = clamp(
                0.74
                + (features["texture_score"] * 0.18)
                + (features["edge_density"] * 0.12)
            )
            return mapped_grade, round(confidence, 2)

        severity = clamp(
            (features["bright_ratio"] * 2.1)
            + (features["edge_density"] * 1.8)
            + (features["texture_score"] * 1.5)
            + (features["asymmetry"] * 2.2)
        )
        if severity < 0.3:
            return "Grade I", round(clamp(0.62 + severity * 0.25, 0.62, 0.9), 2)
        if severity < 0.52:
            return "Grade II", round(clamp(0.66 + severity * 0.23, 0.66, 0.92), 2)
        if severity < 0.72:
            return "Grade III", round(clamp(0.7 + severity * 0.2, 0.7, 0.94), 2)
        return "Grade IV", round(clamp(0.75 + severity * 0.18, 0.75, 0.96), 2)
