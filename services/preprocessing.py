from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import List, Optional, Tuple

import numpy as np

from services.enhancement import ImageEnhancementService
from utils.image_processing import (
    decode_image_bytes,
    ensure_three_channel,
    preprocess_classifier_image_rgb,
    preprocess_segmentation_image,
)
from utils.tensorflow_compat import get_model_manifest_entry

LOGGER = logging.getLogger(__name__)


def _classifier_target_size() -> Tuple[int, int]:
    """Read classifier input size from model manifest (defaults to 150x150)."""
    entry = get_model_manifest_entry("brain_classifier")
    input_shape = entry.get("input_shape")
    if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
        return (int(input_shape[0]), int(input_shape[1]))
    return (150, 150)


@dataclass(frozen=True)
class PreparedAnalysisInputs:
    original_image: np.ndarray
    enhanced_image: np.ndarray
    classifier_input: np.ndarray
    segmentation_input: np.ndarray
    enhancement_backend: str
    enhancement_steps: List[str]


class NeuroImagePreprocessingService:
    """Decode and preprocess scans once for downstream inference services."""

    def __init__(self):
        self._enhancement_service = ImageEnhancementService()

    def prepare(
        self,
        image_bytes: Optional[bytes] = None,
        image: Optional[np.ndarray] = None,
    ) -> PreparedAnalysisInputs:
        if image is None:
            if not image_bytes:
                raise ValueError("Image bytes are required for preprocessing.")
            image = decode_image_bytes(image_bytes)

        original_image = ensure_three_channel(image)
        enhancement_result = self._enhancement_service.enhance(original_image)
        enhanced_image = enhancement_result["enhanced_image"]

        # IMPORTANT: Classification uses the ORIGINAL image (not enhanced)
        # converted to RGB to match ImageDataGenerator training pipelines.
        # Enhancement (CLAHE, sharpening) was NOT used during training,
        # so applying it here would cause distribution shift.
        target_size = _classifier_target_size()
        classifier_input = preprocess_classifier_image_rgb(original_image, target_size=target_size)

        LOGGER.info(
            "Preprocessing complete: classifier_input shape=%s (RGB, no enhancement, target=%s), "
            "segmentation_input from enhanced image",
            classifier_input.shape,
            target_size,
        )

        return PreparedAnalysisInputs(
            original_image=original_image,
            enhanced_image=enhanced_image,
            classifier_input=classifier_input,
            segmentation_input=preprocess_segmentation_image(enhanced_image),
            enhancement_backend=str(enhancement_result.get("backend") or "opencv"),
            enhancement_steps=list(enhancement_result.get("steps") or []),
        )


@lru_cache(maxsize=1)
def get_preprocessing_service() -> NeuroImagePreprocessingService:
    return NeuroImagePreprocessingService()

