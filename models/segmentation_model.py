import os
from typing import Any, Optional

import cv2
import numpy as np

from utils.image_processing import preprocess_segmentation_image
from utils.tensorflow_compat import get_tensorflow_keras, resolve_model_path, safe_load_keras_model


def build_unet_model(
    input_shape: tuple[int, int, int] = (128, 128, 1),
) -> Optional[Any]:
    """Build a U-Net style segmentation network with encoder-decoder skip connections."""
    tensorflow_keras = get_tensorflow_keras()
    if not tensorflow_keras["available"]:
        return None

    layers = tensorflow_keras["layers"]
    models = tensorflow_keras["models"]
    BatchNormalization = layers.BatchNormalization
    Concatenate = layers.Concatenate
    Conv2D = layers.Conv2D
    Dropout = layers.Dropout
    Input = layers.Input
    MaxPooling2D = layers.MaxPooling2D
    UpSampling2D = layers.UpSampling2D
    Model = models.Model

    inputs = Input(shape=input_shape)

    down_1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    down_1 = BatchNormalization()(down_1)
    down_1 = Conv2D(32, (3, 3), activation="relu", padding="same")(down_1)
    down_1 = BatchNormalization()(down_1)
    pool_1 = MaxPooling2D((2, 2))(down_1)

    down_2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool_1)
    down_2 = BatchNormalization()(down_2)
    down_2 = Conv2D(64, (3, 3), activation="relu", padding="same")(down_2)
    down_2 = BatchNormalization()(down_2)
    pool_2 = MaxPooling2D((2, 2))(down_2)

    bridge = Conv2D(128, (3, 3), activation="relu", padding="same")(pool_2)
    bridge = BatchNormalization()(bridge)
    bridge = Conv2D(128, (3, 3), activation="relu", padding="same")(bridge)
    bridge = BatchNormalization()(bridge)
    bridge = Dropout(0.3)(bridge)

    up_1 = UpSampling2D((2, 2))(bridge)
    up_1 = Conv2D(64, (2, 2), activation="relu", padding="same")(up_1)
    up_1 = Concatenate()([up_1, down_2])
    up_1 = Conv2D(64, (3, 3), activation="relu", padding="same")(up_1)
    up_1 = BatchNormalization()(up_1)
    up_1 = Conv2D(64, (3, 3), activation="relu", padding="same")(up_1)
    up_1 = BatchNormalization()(up_1)

    up_2 = UpSampling2D((2, 2))(up_1)
    up_2 = Conv2D(32, (2, 2), activation="relu", padding="same")(up_2)
    up_2 = Concatenate()([up_2, down_1])
    up_2 = Conv2D(32, (3, 3), activation="relu", padding="same")(up_2)
    up_2 = BatchNormalization()(up_2)
    up_2 = Conv2D(32, (3, 3), activation="relu", padding="same")(up_2)
    up_2 = BatchNormalization()(up_2)

    outputs = Conv2D(1, (1, 1), activation="sigmoid", name="tumor_mask")(up_2)
    return Model(inputs=inputs, outputs=outputs, name="tumor_unet")


class TumorSegmentationModel:
    """Lesion segmentation model with heuristic fallback when weights are unavailable."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: tuple[int, int] = (128, 128),
    ):
        default_model_path = model_path or os.path.join("models", "tumor_segmentation_unet.h5")
        self.model_path = (
            resolve_model_path(
                default_model_path,
                "SEGMENTATION_MODEL_PATH",
                manifest_key="tumor_segmentation",
            )
            if default_model_path
            else None
        )
        self.input_size = input_size
        self.model = self._load_model()
        self.backend = "keras" if self.model is not None else "heuristic"

    def _load_model(self):
        if not self.model_path:
            return None
        return safe_load_keras_model(self.model_path)

    def predict_mask(
        self,
        image: np.ndarray,
        tumor_detected: bool = True,
        model_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not tumor_detected:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        model_input = model_input if model_input is not None else preprocess_segmentation_image(image, self.input_size)

        if self.model is not None:
            mask = self._predict_with_model(model_input)
        else:
            mask = self._heuristic_mask(model_input[0, :, :, 0])

        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return self._postprocess_mask(resized_mask)

    def _predict_with_model(self, model_input: np.ndarray) -> np.ndarray:
        try:
            prediction = self.model.predict(model_input, verbose=0)[0, :, :, 0]
            smoothed_prediction = cv2.GaussianBlur(prediction.astype(np.float32), (5, 5), 0)
            return np.where(smoothed_prediction >= 0.45, 255, 0).astype(np.uint8)
        except Exception:
            return self._heuristic_mask(model_input[0, :, :, 0])

    def _heuristic_mask(self, normalized_image: np.ndarray) -> np.ndarray:
        image_u8 = np.clip(normalized_image * 255.0, 0, 255).astype(np.uint8)
        
        # 0. Advanced Skull Stripping
        # Separate the skull from the brain using morphological opening
        _, thresh = cv2.threshold(image_u8, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
        
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            brain_mask = np.zeros_like(image_u8)
            cv2.drawContours(brain_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Erode slightly to remove the bright outer skull/meninges ring
            brain_mask = cv2.erode(brain_mask, kernel, iterations=2)
            image_u8 = cv2.bitwise_and(image_u8, brain_mask)
            
        blurred = cv2.bilateralFilter(image_u8, 9, 75, 75)
        
        # 1. K-Means Clustering to find bright regions
        pixel_values = blurred.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        brightest_cluster = np.argmax(centers)
        mask = (labels == brightest_cluster).reshape(image_u8.shape).astype(np.uint8) * 255
        
        # Fallback if mask is too large or too small
        mask_area = np.count_nonzero(mask)
        total_area = image_u8.shape[0] * image_u8.shape[1]
        if mask_area < total_area * 0.001 or mask_area > total_area * 0.15:
            percentile_value = int(np.percentile(blurred[blurred > 10], 96))
            _, mask = cv2.threshold(blurred, percentile_value, 255, cv2.THRESH_BINARY)
            
        # Light morphological operations to connect nearby pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 2. Find the best tumor component using Solidity and Circularity
        best_component = self._find_best_tumor_blob(mask)
        if np.count_nonzero(best_component) == 0:
            return best_component

        # 3. Apply GrabCut for Pixel-Perfect Boundaries
        try:
            img_3c = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
            # Initialize entire image as probable background so GrabCut can expand the foreground
            grabcut_mask = np.full(image_u8.shape, cv2.GC_PR_BGD, dtype=np.uint8)
            
            # The selected blob is probable foreground
            grabcut_mask[best_component > 0] = cv2.GC_PR_FGD
            
            # Erode the blob to get sure foreground
            sure_fg = cv2.erode(best_component, np.ones((5,5), np.uint8), iterations=2)
            grabcut_mask[sure_fg > 0] = cv2.GC_FGD
            
            # Dilate the blob to restrict GrabCut to a local region. Outside is sure background.
            local_region = cv2.dilate(best_component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)), iterations=1)
            
            # Dark pixels and pixels far from the tumor are sure background
            _, dark_bg = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
            sure_bg = cv2.bitwise_or(dark_bg, cv2.bitwise_not(local_region))
            grabcut_mask[sure_bg > 0] = cv2.GC_BGD

            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            
            cv2.grabCut(img_3c, grabcut_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            final_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            return self._largest_component(final_mask)
        except Exception:
            return best_component

    def _find_best_tumor_blob(self, mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(mask) == 0:
            return np.zeros_like(mask, dtype=np.uint8)

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if component_count <= 1:
            return mask

        best_label = -1
        best_score = -1.0
        
        for i in range(1, component_count):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 30:
                continue
                
            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            contour = contours[0]
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Tumors are usually solid blobs. Skull edges are long/thin (low circularity/solidity)
            score = area * circularity * (solidity ** 2)
            if score > best_score:
                best_score = score
                best_label = i
                
        result = np.zeros_like(mask, dtype=np.uint8)
        if best_label != -1:
            result[labels == best_label] = 255
        return result

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return np.zeros_like(mask, dtype=np.uint8)

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        minimum_area = max(25, int(mask.shape[0] * mask.shape[1] * 0.002))

        if int(stats[largest_label, cv2.CC_STAT_AREA]) < minimum_area:
            return np.zeros_like(mask, dtype=np.uint8)

        return np.where(labels == largest_label, 255, 0).astype(np.uint8)

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        refined = self._refine_mask(mask)
        return self._largest_component(refined)

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        thresholded = cv2.threshold(mask.astype(np.uint8), 32, 255, cv2.THRESH_BINARY)[1]
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((5, 5), np.uint8)

        refined = cv2.erode(thresholded, kernel_small, iterations=1)
        refined = cv2.dilate(refined, kernel_small, iterations=2)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_small, iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        refined = self._largest_component(refined)

        if np.count_nonzero(refined) == 0:
            return refined

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return refined

        filled_mask = np.zeros_like(refined, dtype=np.uint8)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
        return filled_mask
