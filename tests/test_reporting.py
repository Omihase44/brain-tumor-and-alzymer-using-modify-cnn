import base64

import cv2
import numpy as np

from services.reporting import build_medical_report_pdf, build_report_context


def _image_to_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    assert success is True
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def test_build_report_context_and_pdf_include_mask_and_overlay():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[18:46, 18:46] = (180, 180, 180)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:42, 26:44] = 255
    overlay = image.copy()
    overlay[24:42, 26] = (0, 255, 0)

    report = {
        "id": 12,
        "type": "brain",
        "date": "2026-04-11T10:30:00",
        "patient_name": "Jane Doe",
        "patient_age": "47",
        "patient_gender": "Female",
        "patient_id": 4,
        "tumor_type": "glioma tumor",
        "tumor_stage": "Grade III",
        "tumor_confidence": "97.5%",
        "model_accuracy": "97.5%",
        "tumor_volume_mm3": 42.75,
        "doctor_notes": "Recommend prompt neurosurgical review.",
        "prescription": "Dexamethasone 4 mg; Levetiracetam 500 mg",
        "follow_up": "Repeat MRI in 2 weeks; Review with tumor board",
        "report_images": {
            "input_image": _image_to_base64(image),
            "enhanced_image": _image_to_base64(image),
            "mask_image": _image_to_base64(mask),
            "boundary_image": _image_to_base64(overlay),
        },
        "analysis": {
            "tumor": {
                "detected": True,
                "tumor_type": "glioma tumor",
                "tumor_stage": "Grade III",
                "confidence": "97.5%",
                "volume_mm3": 42.75,
            },
            "alzheimers": {
                "stage": "NonDemented",
                "confidence": "12.0%",
            },
            "segmentation": {
                "mask_quality": "clear",
                "contour_count": 1,
                "white_pixel_count": int(np.count_nonzero(mask)),
                "tumor_area_percentage": 7.91,
                "tumor_area_mm2": 18.2,
                "bounding_box": {
                    "width": 18,
                    "height": 18,
                    "width_mm": 18.0,
                    "height_mm": 18.0,
                },
            },
        },
    }

    context = build_report_context(report)

    assert context["patient"]["name"] == "Jane Doe"
    assert len(context["images"]) == 4
    assert context["images"][2]["label"] == "Segmentation Mask"
    assert any(row[0] == "Mask Quality" and row[1] == "Clear" for row in context["segmentation_summary"])

    pdf_buffer = build_medical_report_pdf(report)
    pdf_bytes = pdf_buffer.getvalue()

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000
