import json
import logging
from typing import Dict, Optional

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from services.classification import get_classification_service
from services.enhancement import save_analysis_assets
from services.model_metrics import get_model_metrics
from services.preprocessing import get_preprocessing_service
from services.segmentation import get_segmentation_service
from services.volume_calc import get_volume_calculation_service
from utils.image_processing import (
    decode_base64_image,
    encode_image_to_base64,
    normalize_voxel_metadata,
)
from utils.tensorflow_compat import ModelUnavailableError


analysis_bp = Blueprint("analysis", __name__)
VALID_ANALYSIS_TYPES = {"combined", "brain", "alz"}
LOGGER = logging.getLogger(__name__)


def _format_confidence(confidence: float) -> str:
    confidence_percent = round(float(confidence) * 100, 2)
    formatted = f"{confidence_percent:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def safe_dict(obj) -> Dict[str, object]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped:
            try:
                obj = json.loads(stripped)
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
            return obj if isinstance(obj, dict) else {}
    return {}


def _normalize_confidence_value(value: object) -> float:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("%"):
            normalized = normalized[:-1]
        try:
            value = float(normalized)
        except ValueError:
            return 0.0

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric_value > 1:
        numeric_value /= 100.0
    return numeric_value


def _normalize_detection_type(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"alz", "alzheimer", "alzheimers", "alzheimer's"}:
        return "alz"
    if normalized in {"combined", "both", "all"}:
        return "combined"
    return "brain"


def _resolve_asset_file(files: Dict[str, object], *keys: str) -> Optional[str]:
    for key in keys:
        value = files.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _compute_segmentation_summary(
    mask: np.ndarray,
    voxel_config: Dict[str, float],
    bounding_box: Optional[Dict[str, int]],
    contour_count: int,
) -> Dict[str, object]:
    height, width = mask.shape[:2]
    total_pixels = int(height * width)
    white_pixel_count = int(np.count_nonzero(mask > 0))
    tumor_area_percentage = round((white_pixel_count / total_pixels) * 100.0, 2) if total_pixels else 0.0
    pixel_area_mm2 = round(float(voxel_config["pixel_spacing_x"]) * float(voxel_config["pixel_spacing_y"]), 4)
    tumor_area_mm2 = round(white_pixel_count * pixel_area_mm2, 2)

    bbox_payload = bounding_box or None
    if bbox_payload:
        bbox_payload = {
            "x": int(bbox_payload.get("x") or 0),
            "y": int(bbox_payload.get("y") or 0),
            "width": int(bbox_payload.get("width") or 0),
            "height": int(bbox_payload.get("height") or 0),
            "width_mm": round(int(bbox_payload.get("width") or 0) * float(voxel_config["pixel_spacing_x"]), 2),
            "height_mm": round(int(bbox_payload.get("height") or 0) * float(voxel_config["pixel_spacing_y"]), 2),
        }

    if contour_count and white_pixel_count > 0:
        mask_quality = "clear" if tumor_area_percentage >= 0.05 else "limited"
    elif white_pixel_count > 0:
        mask_quality = "limited"
    else:
        mask_quality = "empty"

    return {
        "white_pixel_count": white_pixel_count,
        "tumor_area_percentage": tumor_area_percentage,
        "pixel_area_mm2": pixel_area_mm2,
        "tumor_area_mm2": tumor_area_mm2,
        "bounding_box": bbox_payload,
        "mask_quality": mask_quality,
    }


def _resolve_accuracy_label(value: object, metrics: object) -> str:
    metrics_payload = safe_dict(metrics)
    accuracy_value = value if value not in (None, "") else metrics_payload.get("accuracy") or metrics_payload.get("accuracy_label")
    if accuracy_value in (None, ""):
        return "Unavailable"
    if isinstance(accuracy_value, str):
        normalized = accuracy_value.strip()
        if normalized.endswith("%"):
            return normalized
        try:
            accuracy_value = float(normalized)
        except ValueError:
            return normalized
    numeric_value = float(accuracy_value)
    if numeric_value <= 1:
        numeric_value *= 100.0
    return f"{numeric_value:.2f}".rstrip("0").rstrip(".") + "%"


def _build_filtered_ai_result(analysis: Dict[str, object], detection_type: str) -> Dict[str, object]:
    tumor_payload = safe_dict(analysis.get("tumor"))
    alzheimers_payload = safe_dict(analysis.get("alzheimers"))
    tumor_result = {
        "type": "Brain Tumor",
        "result": "Detected" if tumor_payload.get("detected") else "Not Detected",
        "tumor_type": tumor_payload.get("tumor_type") or tumor_payload.get("classification") or "No Tumor",
        "tumor_stage": tumor_payload.get("tumor_stage") or tumor_payload.get("grade") or "N/A",
        "confidence": tumor_payload.get("confidence") or "N/A",
        "model_accuracy": _resolve_accuracy_label(analysis.get("model_accuracy"), tumor_payload.get("model_metrics")),
    }
    alzheimer_result = {
        "type": "Alzheimer",
        "stage": alzheimers_payload.get("stage") or analysis.get("alzheimer_stage") or "NonDemented",
        "confidence": alzheimers_payload.get("confidence") or "N/A",
        "model_accuracy": _resolve_accuracy_label(
            safe_dict(alzheimers_payload.get("model_metrics")).get("accuracy_label") or analysis.get("model_accuracy"),
            alzheimers_payload.get("model_metrics"),
        ),
    }

    if detection_type == "alz":
        return alzheimer_result
    if detection_type == "combined":
        return {"brain_tumor": tumor_result, "alzheimer": alzheimer_result}
    return tumor_result


def _normalize_tumor_prediction(value: object) -> Dict[str, object]:
    result = safe_dict(value)
    detected = bool(result.get("detected", False))
    return {
        "detected": detected,
        "classification": result.get("classification") or ("Tumor Detected" if detected else "No Tumor"),
        "tumor_type": result.get("tumor_type") or result.get("classification") or ("Tumor Detected" if detected else "No Tumor"),
        "grade": result.get("grade"),
        "tumor_stage": result.get("tumor_stage") or result.get("grade"),
        "confidence": _normalize_confidence_value(result.get("confidence", 0)),
        "type_confidence": _normalize_confidence_value(result.get("type_confidence", result.get("confidence", 0))),
        "stage_confidence": _normalize_confidence_value(result.get("stage_confidence", result.get("confidence", 0))),
        "model_metrics": safe_dict(result.get("model_metrics")),
    }


def _normalize_alzheimers_prediction(value: object) -> Dict[str, object]:
    result = safe_dict(value)
    detected = bool(result.get("detected", False))
    return {
        "detected": detected,
        "stage": result.get("stage") or ("Moderate" if detected else "NonDemented"),
        "confidence": _normalize_confidence_value(result.get("confidence", 0)),
        "model_metrics": safe_dict(result.get("model_metrics")),
    }


def predict_tumor(image: np.ndarray) -> Dict[str, object]:
    classification_service = get_classification_service()
    classification_result = safe_dict(classification_service.classify(image, detection_type="brain"))
    result = _normalize_tumor_prediction(classification_result.get("tumor"))
    return {
        "detected": result["detected"],
        "classification": result["classification"],
        "tumor_type": result["tumor_type"],
        "grade": result["grade"],
        "tumor_stage": result["tumor_stage"],
        "confidence": result["confidence"],
        "model_metrics": result["model_metrics"] or get_model_metrics("brain_classifier"),
    }


def analyze_medical_image(
    image_bytes: bytes,
    detection_type: str = "combined",
    voxel_metadata: Optional[Dict[str, object]] = None,
    include_encoded_images: bool = True,
) -> Dict[str, object]:
    """Run the unified neurodiagnostic pipeline for a single MRI image."""
    preprocessing_service = get_preprocessing_service()
    classification_service = get_classification_service()
    segmentation_service = get_segmentation_service()
    volume_service = get_volume_calculation_service()

    prepared_inputs = preprocessing_service.prepare(image_bytes=image_bytes)
    original_image = prepared_inputs.original_image
    enhanced_image = prepared_inputs.enhanced_image
    classification_result = safe_dict(
        classification_service.classify(
            enhanced_image,
            detection_type=detection_type,
            prepared_inputs=prepared_inputs,
        )
    )
    tumor_result = _normalize_tumor_prediction(classification_result.get("tumor"))
    alzheimer_result = _normalize_alzheimers_prediction(classification_result.get("alzheimers"))
    segmentation_result = segmentation_service.segment(
        original_image=enhanced_image,
        working_image=enhanced_image,
        tumor_detected=tumor_result["detected"],
        prepared_input=prepared_inputs.segmentation_input,
    )
    contour_mask = segmentation_result["mask"]

    voxel_config = normalize_voxel_metadata(voxel_metadata)
    volume_result = volume_service.calculate(
        contour_mask,
        pixel_spacing_x=voxel_config["pixel_spacing_x"],
        pixel_spacing_y=voxel_config["pixel_spacing_y"],
        slice_thickness=voxel_config["slice_thickness"],
    )

    if not tumor_result["detected"]:
        volume_result["tumor_volume_mm3"] = 0.0
        volume_result["white_pixel_count"] = 0
        contour_mask[:, :] = 0

    boundary_overlay = segmentation_result["overlay"]
    assets = save_analysis_assets(
        {
            "input_image": original_image,
            "enhanced_image": enhanced_image,
            "mask": contour_mask,
            "boundary_overlay": boundary_overlay,
        }
    )
    asset_files = assets.get("files") if isinstance(assets.get("files"), dict) else {}
    input_image_path = _resolve_asset_file(asset_files, "input_image", "original_image")
    enhanced_image_path = _resolve_asset_file(asset_files, "enhanced_image")
    mask_image_path = _resolve_asset_file(asset_files, "mask", "mask_image", "segmentation_mask", "display_mask")
    boundary_image_path = _resolve_asset_file(asset_files, "boundary_overlay", "boundary_image", "overlay_image", "segmentation_overlay")

    original_base64 = encode_image_to_base64(original_image, ".png") if include_encoded_images else None
    enhanced_base64 = encode_image_to_base64(enhanced_image, ".png") if include_encoded_images else None
    mask_base64 = encode_image_to_base64(contour_mask, ".png") if include_encoded_images else None
    boundary_base64 = encode_image_to_base64(boundary_overlay, ".png") if include_encoded_images else None
    tumor_volume_mm3 = round(volume_result["tumor_volume_mm3"], 2)
    detection_type = (detection_type or "combined").lower()
    tumor_metrics = tumor_result.get("model_metrics") or get_model_metrics("brain_classifier")
    alzheimer_metrics = alzheimer_result.get("model_metrics") or get_model_metrics("alzheimer_classifier")
    segmentation_summary = _compute_segmentation_summary(
        contour_mask,
        voxel_config=voxel_config,
        bounding_box=segmentation_result["bounding_box"],
        contour_count=int(segmentation_result["contour_count"] or 0),
    )

    tumor_payload = {
        **tumor_result,
        "tumor_type": tumor_result["tumor_type"],
        "tumor_stage": tumor_result["tumor_stage"],
        "confidence_score": round(float(tumor_result["confidence"]), 4),
        "confidence": _format_confidence(tumor_result["confidence"]),
        "type_confidence_score": round(float(tumor_result["type_confidence"]), 4),
        "type_confidence": _format_confidence(tumor_result["type_confidence"]),
        "stage_confidence_score": round(float(tumor_result["stage_confidence"]), 4),
        "stage_confidence": _format_confidence(tumor_result["stage_confidence"]),
        "model_metrics": tumor_metrics,
        "volume_mm3": tumor_volume_mm3,
        "tumor_volume_mm3": tumor_volume_mm3,
    }
    alzheimer_payload = {
        **alzheimer_result,
        "confidence_score": round(float(alzheimer_result["confidence"]), 4),
        "confidence": _format_confidence(alzheimer_result["confidence"]),
        "model_metrics": alzheimer_metrics,
    }
    tumor_accuracy_label = tumor_metrics.get("accuracy_label") or "Unavailable"
    tumor_accuracy_score = tumor_metrics.get("accuracy")
    alzheimer_accuracy_label = alzheimer_metrics.get("accuracy_label") or "Unavailable"

    response = {
        "analysis_type": detection_type,
        "tumor": tumor_payload,
        "alzheimers": alzheimer_payload,
        "model_metrics": {
            "tumor": tumor_metrics,
            "alzheimers": alzheimer_metrics,
        },
        "enhancement": {
            "backend": prepared_inputs.enhancement_backend,
            "steps": prepared_inputs.enhancement_steps,
            "input_image": input_image_path,
            "enhanced_image": enhanced_image_path,
            "input_image_base64": original_base64,
            "enhanced_image_base64": enhanced_base64,
        },
        "segmentation": {
            "available": bool(tumor_result["detected"]),
            "backend": segmentation_result["backend"],
            "white_pixel_count": segmentation_summary["white_pixel_count"],
            "voxel_volume_mm3": volume_result["voxel_volume_mm3"],
            "tumor_area_percentage": segmentation_summary["tumor_area_percentage"],
            "tumor_area_mm2": segmentation_summary["tumor_area_mm2"],
            "pixel_area_mm2": segmentation_summary["pixel_area_mm2"],
            "mask_quality": segmentation_summary["mask_quality"],
            "mask_image": mask_image_path,
            "mask_image_base64": mask_base64,
            "boundary_image": boundary_image_path,
            "segmentation_overlay": boundary_image_path,
            "boundary_image_base64": boundary_base64,
            "bounding_box": segmentation_summary["bounding_box"],
            "contour_count": segmentation_result["contour_count"],
        },
        "images": {
            "input": {"path": input_image_path, "base64": original_base64},
            "enhanced": {"path": enhanced_image_path, "base64": enhanced_base64},
            "mask": {"path": mask_image_path, "base64": mask_base64},
            "boundary": {"path": boundary_image_path, "base64": boundary_base64},
        },
        "study_id": assets["study_id"],
        "input_image": input_image_path,
        "enhanced_image": enhanced_image_path,
        "mask_image": mask_image_path,
        "boundary_image": boundary_image_path,
        "segmentation_overlay": boundary_image_path,
        "input_image_base64": original_base64,
        "enhanced_image_base64": enhanced_base64,
        "mask_image_base64": mask_base64,
        "boundary_image_base64": boundary_base64,
        "voxel_metadata": voxel_config,
        "white_pixel_count": segmentation_summary["white_pixel_count"],
        "tumor_area_percentage": segmentation_summary["tumor_area_percentage"],
        "tumor_area_mm2": segmentation_summary["tumor_area_mm2"],
        "tumor_detected": tumor_result["detected"],
        "tumor_type": tumor_payload["tumor_type"],
        "tumor_grade": tumor_result["grade"],
        "tumor_stage": tumor_payload["tumor_stage"],
        "tumor_confidence": tumor_payload["confidence"],
        "tumor_confidence_score": round(float(tumor_result["confidence"]), 4),
        "model_accuracy": tumor_accuracy_label,
        "model_accuracy_score": tumor_accuracy_score,
        "tumor_volume_mm3": tumor_volume_mm3,
        "alzheimers_detected": alzheimer_result["detected"],
        "alz_detected": alzheimer_result["detected"],
        "alzheimer_stage": alzheimer_result["stage"],
        "alz_stage": alzheimer_result["stage"],
        "alzheimer_confidence": alzheimer_payload["confidence"],
        "alzheimer_confidence_score": round(float(alzheimer_result["confidence"]), 4),
        "ai_clinical_insights": {
            "tumor_type": tumor_payload["tumor_type"],
            "tumor_stage": tumor_payload["tumor_stage"],
            "tumor_confidence": tumor_payload["confidence"],
            "tumor_volume_mm3": tumor_volume_mm3,
            "tumor_area_percentage": segmentation_summary["tumor_area_percentage"],
            "tumor_area_mm2": segmentation_summary["tumor_area_mm2"],
            "model_accuracy": tumor_accuracy_label,
            "model_accuracy_score": tumor_accuracy_score,
            "alzheimer_stage": alzheimer_result["stage"],
            "alzheimer_model_accuracy": alzheimer_accuracy_label,
            "boundary_status": "Detected" if segmentation_result["contour_count"] else "Not detected",
        },
        "confidence": _format_confidence(
            max(float(tumor_result["confidence"]), float(alzheimer_result["confidence"]))
        ),
    }

    if detection_type == "brain":
        response["primary_result"] = {
            "type": "tumor",
            "detected": tumor_payload["detected"],
            "classification": tumor_payload["classification"],
            "tumor_type": tumor_payload["tumor_type"],
            "grade": tumor_payload["grade"],
            "tumor_stage": tumor_payload["tumor_stage"],
            "confidence": tumor_payload["confidence"],
        }
    elif detection_type == "alz":
        response["primary_result"] = {
            "type": "alzheimers",
            "detected": alzheimer_payload["detected"],
            "stage": alzheimer_payload["stage"],
            "confidence": alzheimer_payload["confidence"],
        }
    else:
        response["primary_result"] = {
            "type": "combined",
            "tumor_detected": tumor_payload["detected"],
            "alzheimers_detected": alzheimer_payload["detected"],
        }

    return response


def _extract_request_image_bytes() -> bytes:
    for field_name in ("file", "image", "scan"):
        if field_name in request.files:
            uploaded_file = request.files[field_name]
            image_bytes = uploaded_file.read()
            if not image_bytes:
                raise ValueError("Uploaded image file is empty.")
            return image_bytes

    payload = _extract_json_payload()
    image_base64 = payload.get("image") or payload.get("image_base64")
    if image_base64:
        image = decode_base64_image(image_base64)
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Unable to decode the submitted base64 image.")
        return encoded.tobytes()

    raise ValueError("No image provided. Use multipart file upload or base64 JSON.")


def _extract_voxel_payload() -> Dict[str, object]:
    if request.is_json:
        payload = _extract_json_payload()
        voxel_payload = _coerce_mapping(payload.get("voxel_metadata"), allow_raw_image=False)
        return {
            "pixel_spacing_x": payload.get("pixel_spacing_x", voxel_payload.get("pixel_spacing_x")),
            "pixel_spacing_y": payload.get("pixel_spacing_y", voxel_payload.get("pixel_spacing_y")),
            "slice_thickness": payload.get("slice_thickness", voxel_payload.get("slice_thickness")),
        }

    return {
        "pixel_spacing_x": request.form.get("pixel_spacing_x"),
        "pixel_spacing_y": request.form.get("pixel_spacing_y"),
        "slice_thickness": request.form.get("slice_thickness"),
    }


def _coerce_mapping(value, allow_raw_image: bool = False) -> Dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"image": stripped} if allow_raw_image else {}
        if isinstance(parsed, dict):
            return parsed
        if allow_raw_image and isinstance(parsed, str):
            return {"image": parsed}
    return {}


def _extract_json_payload() -> Dict[str, object]:
    return safe_dict(_coerce_mapping(request.get_json(silent=True), allow_raw_image=True))


@analysis_bp.route("/analyze", methods=["POST"])
@analysis_bp.route("/api/analyze", methods=["POST"])
def analyze_route():
    try:
        configured_api_keys = current_app.config.get("API_KEYS") or set()
        if configured_api_keys and "user_id" not in request.headers and "user_id" not in request.form:
            api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
            if api_key not in configured_api_keys:
                return jsonify({"success": False, "error": "Unauthorized API request."}), 401

        detection_type = _normalize_detection_type(
            request.form.get("type") or request.args.get("type") or _extract_json_payload().get("type") or "brain"
        )
        image_bytes = _extract_request_image_bytes()
        analysis = analyze_medical_image(
            image_bytes=image_bytes,
            detection_type=detection_type,
            voxel_metadata=_extract_voxel_payload(),
            include_encoded_images=True,
        )
        images_payload = safe_dict(analysis.get("images"))

        return jsonify(
            {
                "success": True,
                "input_image": safe_dict(images_payload.get("input")).get("base64"),
                "enhanced_image": safe_dict(images_payload.get("enhanced")).get("base64"),
                "mask_image": safe_dict(images_payload.get("mask")).get("base64"),
                "boundary_image": safe_dict(images_payload.get("boundary")).get("base64"),
                "segmentation": safe_dict(analysis.get("segmentation")),
                "ai_result": _build_filtered_ai_result(analysis, detection_type),
            }
        )
    except ModelUnavailableError as exc:
        LOGGER.error("Model unavailable for analysis request: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        LOGGER.exception("Unexpected failure during /analyze request")
        return jsonify({"success": False, "error": str(exc)}), 500
