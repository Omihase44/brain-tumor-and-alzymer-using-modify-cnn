from functools import lru_cache
from typing import Dict, Optional

import cv2
import numpy as np

from services.model_registry import get_model_registry
from utils.image_processing import ensure_three_channel


class SegmentationService:
    """Generate a stable single-region tumor mask and clean boundary overlay."""

    def __init__(self):
        self._model_registry = get_model_registry()
        self.model = None

    def segment(
        self,
        original_image: np.ndarray,
        working_image: np.ndarray,
        tumor_detected: bool = True,
        prepared_input: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        original = ensure_three_channel(original_image)
        working = ensure_three_channel(working_image)

        if not tumor_detected:
            empty_mask = np.zeros(original.shape[:2], dtype=np.uint8)
            return {
                "mask": empty_mask,
                "overlay": original.copy(),
                "bounding_box": None,
                "backend": "empty_mask_no_tumor",
                "contour_count": 0,
            }

        candidate_image = self._preprocess_candidate_image(working)
        brain_mask = self._build_brain_mask(original)
        segmentation_model = self._get_segmentation_model()

        model_mask = segmentation_model.predict_mask(
            working,
            tumor_detected=tumor_detected,
            model_input=prepared_input,
        )
        refined_mask = self._postprocess_mask(
            model_mask,
            candidate_image=candidate_image,
            brain_mask=brain_mask,
        )

        backend = f"{getattr(segmentation_model, 'backend', 'model')}_postprocessed"
        if np.count_nonzero(refined_mask) == 0:
            heuristic_mask = self._build_high_intensity_mask(candidate_image, brain_mask)
            refined_mask = self._postprocess_mask(
                heuristic_mask,
                candidate_image=candidate_image,
                brain_mask=brain_mask,
            )
            backend = "opencv_percentile97_postprocessed"

        contour = self._extract_contour(refined_mask)
        overlay = self._draw_overlay(original, contour)
        bounding_box = self._get_bbox(refined_mask)

        return {
            "mask": refined_mask,
            "display_mask": refined_mask.copy(),
            "overlay": overlay,
            "bounding_box": bounding_box,
            "backend": backend,
            "contour_count": 1 if contour is not None else 0,
        }

    def _get_segmentation_model(self):
        injected_model = getattr(self, "model", None)
        if injected_model is not None:
            return injected_model

        model_registry = getattr(self, "_model_registry", None)
        if model_registry is None:
            model_registry = get_model_registry()
            self._model_registry = model_registry
        return model_registry.get_segmentation_model()

    def _preprocess_candidate_image(self, image: np.ndarray) -> np.ndarray:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        return cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

    def _build_brain_mask(self, image: np.ndarray) -> np.ndarray:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(grayscale, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones_like(grayscale, dtype=np.uint8) * 255

        largest_contour = max(contours, key=cv2.contourArea)
        brain_mask = np.zeros_like(grayscale, dtype=np.uint8)
        cv2.drawContours(brain_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Cut off the bottom 15% of the brain mask bounding box to remove the neck/jaw area
        x, y, w, h = cv2.boundingRect(largest_contour)
        cutoff_y = int(y + h * 0.85)
        brain_mask[cutoff_y:, :] = 0
        
        brain_mask = cv2.morphologyEx(
            brain_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=2,
        )
        brain_mask = cv2.erode(
            brain_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=3,
        )
        return brain_mask

    def _build_high_intensity_mask(self, image: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        high_intensity_mask = np.zeros_like(image, dtype=np.uint8)
        
        # Apply bilateral filter to smooth noise
        blurred = cv2.bilateralFilter(image, 9, 75, 75)
        inside_brain = blurred[brain_mask > 0]
        
        if inside_brain.size < 100:
            return high_intensity_mask

        # Apply k-means (k=3) on the brain pixels
        pixel_values = inside_brain.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        brightest_cluster = np.argmax(centers)
        threshold_value = float(np.min(pixel_values[labels == brightest_cluster]))
        
        # Additional safety to prevent choosing background
        if threshold_value < 50:
            threshold_value = float(np.percentile(inside_brain, 95.0))
            
        high_intensity_mask[(blurred >= threshold_value) & (brain_mask > 0)] = 255
        return high_intensity_mask

    def _postprocess_mask(
        self,
        predicted_mask: np.ndarray,
        candidate_image: np.ndarray,
        brain_mask: np.ndarray,
    ) -> np.ndarray:
        binary_mask = self._to_binary_mask(predicted_mask)
        binary_mask = cv2.bitwise_and(binary_mask, brain_mask)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

        best_region = self._select_best_region(binary_mask, candidate_image)
        if np.count_nonzero(best_region) == 0:
            return best_region

        smoothed_mask = self._smooth_mask(best_region)
        smoothed_mask = cv2.bitwise_and(smoothed_mask, brain_mask)
        return self._largest_component(smoothed_mask)

    def _to_binary_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.dtype.kind in {"f"}:
            return np.where(mask >= 0.5, 255, 0).astype(np.uint8)
        return np.where(mask >= 127, 255, 0).astype(np.uint8)

    def _select_best_region(self, mask: np.ndarray, candidate_image: np.ndarray) -> np.ndarray:
        if np.count_nonzero(mask) == 0:
            return np.zeros_like(mask, dtype=np.uint8)

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            return self._validate_single_region(mask)

        height, width = mask.shape
        image_area = float(height * width)
        best_label = 0
        best_score = -np.inf

        for label_index in range(1, component_count):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < 200 or area > int(image_area * 0.10):
                continue

            x = int(stats[label_index, cv2.CC_STAT_LEFT])
            y = int(stats[label_index, cv2.CC_STAT_TOP])
            region_width = int(stats[label_index, cv2.CC_STAT_WIDTH])
            region_height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            touches_full_border = (
                x <= 2
                and y <= 2
                and (x + region_width) >= (width - 2)
                and (y + region_height) >= (height - 2)
            )
            if touches_full_border:
                continue

            component_mask = np.zeros_like(mask, dtype=np.uint8)
            component_mask[labels == label_index] = 255
            mean_intensity = float(cv2.mean(candidate_image, mask=component_mask)[0])
            area_ratio = area / image_area
            size_score = 1.0 - min(abs(area_ratio - 0.02) / 0.02, 1.0)
            score = (mean_intensity / 255.0) * 2.0 + size_score + min(area_ratio / 0.03, 1.0)

            if score > best_score:
                best_score = score
                best_label = label_index

        if best_label == 0:
            return np.zeros_like(mask, dtype=np.uint8)

        best_region = np.zeros_like(mask, dtype=np.uint8)
        best_region[labels == best_label] = 255
        return best_region

    def _validate_single_region(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask, dtype=np.uint8)

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        image_area = float(mask.shape[0] * mask.shape[1])
        x, y, width, height = cv2.boundingRect(contour)
        touches_full_border = (
            x <= 2
            and y <= 2
            and (x + width) >= (mask.shape[1] - 2)
            and (y + height) >= (mask.shape[0] - 2)
        )
        if area < 200.0 or area > image_area * 0.10 or touches_full_border:
            return np.zeros_like(mask, dtype=np.uint8)

        validated = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(validated, [contour], -1, 255, thickness=cv2.FILLED)
        return validated

    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(mask, (9, 9), 0)
        _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        thresholded = cv2.morphologyEx(
            thresholded,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        thresholded = cv2.erode(
            thresholded,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        thresholded = cv2.dilate(
            thresholded,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        return thresholded

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(mask) == 0:
            return np.zeros_like(mask, dtype=np.uint8)

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            return mask.copy()

        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_mask = np.zeros_like(mask, dtype=np.uint8)
        largest_mask[labels == largest_label] = 255
        return largest_mask

    def _extract_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        return self._smooth_contour(contour)

    def _smooth_contour(self, contour: np.ndarray) -> Optional[np.ndarray]:
        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 3:
            return None
            
        # Use a very light polygon approximation to smooth jagged edges 
        # without ballooning or distorting the pixel-perfect GrabCut boundary
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx.astype(np.int32)

    def _draw_overlay(self, image: np.ndarray, contour: Optional[np.ndarray]) -> np.ndarray:
        overlay = image.copy()
        if contour is None or len(contour) == 0:
            return overlay

        cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        return overlay

    def _get_bbox(self, mask: np.ndarray) -> Optional[Dict[str, int]]:
        if np.count_nonzero(mask) == 0:
            return None

        x, y, width, height = cv2.boundingRect(mask)
        return {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        }


@lru_cache(maxsize=1)
def get_segmentation_service():
    return SegmentationService()
