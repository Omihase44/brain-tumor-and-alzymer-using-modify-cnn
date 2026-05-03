import base64
import io
import os
from datetime import datetime
from typing import Dict, List, Optional

from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
PRIMARY_BLUE = colors.HexColor("#0E5A7A")
PRIMARY_BLUE_DARK = colors.HexColor("#083A4F")
SOFT_BLUE = colors.HexColor("#EDF6FA")
SURFACE = colors.HexColor("#F8FBFC")
BORDER = colors.HexColor("#D4E2E8")
TEXT = colors.HexColor("#1F2F38")
MUTED = colors.HexColor("#637885")
ALERT = colors.HexColor("#FFF7EC")


def _safe_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_text(value, default: str = "N/A") -> str:
    if value in (None, ""):
        return default
    return str(value)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_percent(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        normalized = str(value).strip().replace("%", "")
        if not normalized:
            return None
        numeric = float(normalized)
    except (TypeError, ValueError):
        return None
    if numeric <= 1:
        numeric *= 100.0
    return numeric


def _format_mm(value: Optional[float]) -> str:
    if value in (None, ""):
        return "Unavailable"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    formatted = f"{numeric:.2f}".rstrip("0").rstrip(".")
    return f"{formatted} mm"


def _format_mm2(value: Optional[float]) -> str:
    if value in (None, ""):
        return "Unavailable"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    formatted = f"{numeric:.2f}".rstrip("0").rstrip(".")
    return f"{formatted} mm2"


def _format_mm3(value: Optional[float]) -> str:
    if value in (None, ""):
        return "Unavailable"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    formatted = f"{numeric:.2f}".rstrip("0").rstrip(".")
    return f"{formatted} mm3"


def _safe_image_value(value) -> Optional[str]:
    if isinstance(value, dict):
        return (
            value.get("base64")
            or value.get("image")
            or value.get("image_base64")
            or value.get("data")
            or value.get("path")
        )
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _split_bullets(value: Optional[str], fallback: str) -> List[str]:
    text = str(value or fallback).replace("Ã¢â‚¬Â¢", "\n").replace("â€¢", "\n").replace(";", "\n")
    items = [item.strip() for item in text.splitlines() if item.strip()]
    return items or [fallback]


def _format_report_date(value: object) -> str:
    raw_value = _safe_text(value, "")
    if not raw_value:
        return "N/A"
    for parser in (datetime.fromisoformat,):
        try:
            parsed = parser(raw_value)
            return parsed.strftime("%d %b %Y, %I:%M %p")
        except Exception:
            continue
    return raw_value


def _open_image_stream(image_value: Optional[str]) -> Optional[io.BytesIO]:
    if not image_value:
        return None

    normalized = str(image_value).strip()
    if not normalized:
        return None

    if os.path.exists(normalized):
        try:
            with open(normalized, "rb") as file_handle:
                return io.BytesIO(file_handle.read())
        except Exception:
            return None

    if "," in normalized:
        normalized = normalized.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(normalized)
    except Exception:
        return None
    return io.BytesIO(image_bytes)


def _prepare_report_image_stream(image_value: Optional[str], sharpen: bool = False) -> Optional[io.BytesIO]:
    image_stream = _open_image_stream(image_value)
    if image_stream is None:
        return None

    try:
        image = Image.open(image_stream).convert("RGB")
    except Exception:
        return None

    image.thumbnail((1800, 1800), RESAMPLE_LANCZOS)
    if sharpen:
        image = image.filter(ImageFilter.SHARPEN)

    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG", optimize=True)
    output_buffer.seek(0)
    return output_buffer


def _to_image_src(image_value: Optional[str]) -> str:
    if not image_value:
        return ""
    normalized = str(image_value).strip()
    if not normalized:
        return ""
    if normalized.startswith("data:") or normalized.startswith("http://") or normalized.startswith("https://"):
        return normalized
    if os.path.exists(normalized):
        try:
            with open(normalized, "rb") as file_handle:
                encoded = base64.b64encode(file_handle.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception:
            return ""
    return f"data:image/png;base64,{normalized}"


def _resolve_detection_type(report: dict) -> str:
    raw_type = str(report.get("type") or "").strip().lower()
    if raw_type in {"brain", "tumor"}:
        return "Brain Tumor"
    if raw_type in {"alz", "alzheimer", "alzheimers", "alzheimer's"}:
        return "Alzheimer's"
    if raw_type in {"combined", "both"}:
        return "Combined Analysis"
    return raw_type.title() or "Brain Tumor"


def _resolve_report_image(report: dict, report_images: dict, analysis: dict, segmentation: dict, *keys: str) -> Optional[str]:
    analysis_images = _safe_dict(analysis.get("images"))
    candidates: List[object] = []

    for key in keys:
        candidates.extend(
            [
                report_images.get(key),
                report.get(key),
                analysis.get(key),
                segmentation.get(key),
            ]
        )

    image_aliases = {
        "input_image": ("input", "original"),
        "enhanced_image": ("enhanced",),
        "mask_image": ("mask",),
        "boundary_image": ("boundary", "overlay"),
    }
    for key in keys:
        for alias in image_aliases.get(key, ()):
            candidates.append(_safe_dict(analysis_images.get(alias)))

    for candidate in candidates:
        resolved = _safe_image_value(candidate)
        if resolved:
            return resolved
    return None


def build_report_context(report: dict) -> Dict[str, object]:
    report = dict(report or {})
    analysis = _safe_dict(report.get("analysis"))
    tumor = _safe_dict(analysis.get("tumor"))
    alzheimers = _safe_dict(analysis.get("alzheimers"))
    segmentation = _safe_dict(analysis.get("segmentation"))
    report_images = _safe_dict(report.get("report_images"))
    bounding_box = _safe_dict(segmentation.get("bounding_box"))

    input_image = _resolve_report_image(report, report_images, analysis, segmentation, "input_image", "image", "original_image", "original_mri")
    enhanced_image = _resolve_report_image(report, report_images, analysis, segmentation, "enhanced_image", "enhanced_mri")
    mask_image = _resolve_report_image(report, report_images, analysis, segmentation, "mask_image", "segmentation_mask")
    boundary_image = _resolve_report_image(report, report_images, analysis, segmentation, "boundary_image", "segmentation_overlay", "segmentation_image", "segmented_image")

    white_pixel_count = int(segmentation.get("white_pixel_count") or report.get("white_pixel_count") or 0)
    tumor_area_percentage = _parse_percent(
        segmentation.get("tumor_area_percentage") or report.get("tumor_area_percentage")
    )
    if tumor_area_percentage is None:
        tumor_area_percentage = 0.0
    tumor_area_mm2 = _safe_float(
        segmentation.get("tumor_area_mm2") or report.get("tumor_area_mm2"),
        0.0,
    )
    tumor_volume_mm3 = _safe_float(
        tumor.get("volume_mm3") or report.get("tumor_volume_mm3"),
        0.0,
    )
    contour_count = int(segmentation.get("contour_count") or 0)
    mask_quality = _safe_text(segmentation.get("mask_quality"), "Unavailable")

    detection_status = "Detected" if bool(tumor.get("detected", report.get("tumor_detected"))) else "Not Detected"
    tumor_type = _safe_text(report.get("tumor_type") or tumor.get("tumor_type") or tumor.get("classification"))
    tumor_stage = _safe_text(report.get("tumor_stage") or report.get("tumor_grade") or tumor.get("tumor_stage") or tumor.get("grade"))
    tumor_confidence = _safe_text(report.get("tumor_confidence") or tumor.get("confidence"))
    alzheimer_stage = _safe_text(report.get("alzheimer_stage") or report.get("alz_stage") or alzheimers.get("stage"))
    alzheimer_confidence = _safe_text(report.get("alzheimer_confidence") or alzheimers.get("confidence"))
    model_accuracy = _safe_text(report.get("model_accuracy") or analysis.get("model_accuracy"), "Unavailable")

    size_label = "No measurable tumor"
    if white_pixel_count > 0:
        if tumor_area_percentage >= 3.0:
            size_label = "Large"
        elif tumor_area_percentage >= 1.0:
            size_label = "Medium"
        elif tumor_area_percentage > 0:
            size_label = "Small"

    if detection_status == "Detected":
        interpretation = (
            f"AI analysis identified a {size_label.lower()} focal lesion pattern consistent with {tumor_type.lower()}."
            f" Estimated tumor occupancy is {tumor_area_percentage:.2f}% of the image field."
            " Findings should be correlated with the complete radiology study and clinical assessment."
        )
    elif _resolve_detection_type(report) == "Alzheimer's":
        interpretation = (
            f"AI staging suggests {alzheimer_stage.lower()} neurocognitive change."
            " This output is supportive only and requires formal neurological correlation."
        )
    else:
        interpretation = "No stable lesion contour was retained after segmentation cleanup. Clinical review remains recommended if symptoms persist."

    prescription_items = _split_bullets(
        report.get("prescription"),
        "Treatment recommendations must be finalized by the reviewing clinician.",
    )
    follow_up_items = _split_bullets(
        report.get("follow_up"),
        "Schedule clinical follow-up and repeat imaging as advised by the treating team.",
    )

    bbox_width = int(bounding_box.get("width") or 0)
    bbox_height = int(bounding_box.get("height") or 0)
    bbox_width_mm = _safe_float(bounding_box.get("width_mm"), 0.0)
    bbox_height_mm = _safe_float(bounding_box.get("height_mm"), 0.0)

    return {
        "system_name": "NeuroDetect AI",
        "system_subtitle": "AI-assisted neuroimaging decision support",
        "report_title": "Clinical Imaging Report",
        "report_id": _safe_text(report.get("id")),
        "report_date": _format_report_date(report.get("report_created_at") or report.get("date")),
        "patient": {
            "name": _safe_text(report.get("patient_name")),
            "age": _safe_text(report.get("patient_age")),
            "gender": _safe_text(report.get("patient_gender")),
            "patient_id": _safe_text(report.get("patient_id")),
        },
        "scan_details": [
            ("Detection Type", _resolve_detection_type(report)),
            ("Study ID", _safe_text(report.get("study_id"), "Unavailable")),
            ("Model Accuracy", model_accuracy),
            ("Status", _safe_text(report.get("status"), "Pending").title()),
        ],
        "ai_summary": [
            ("Detection Status", detection_status),
            ("Tumor Type", tumor_type),
            ("Tumor Stage", tumor_stage),
            ("Tumor Confidence", tumor_confidence),
            ("Alzheimer Stage", alzheimer_stage),
            ("Alzheimer Confidence", alzheimer_confidence),
        ],
        "segmentation_summary": [
            ("Mask Quality", mask_quality.title()),
            ("Contour Count", str(contour_count)),
            ("Tumor Area", f"{tumor_area_percentage:.2f}%" if tumor_area_percentage else "0%"),
            ("Tumor Area (mm2)", _format_mm2(tumor_area_mm2) if tumor_area_mm2 else "Unavailable"),
            ("Tumor Volume", _format_mm3(tumor_volume_mm3) if tumor_volume_mm3 else "Unavailable"),
            (
                "Bounding Box",
                f"{bbox_width} x {bbox_height} px" if bbox_width and bbox_height else "Unavailable",
            ),
            (
                "Bounding Box (mm)",
                f"{_format_mm(bbox_width_mm)} x {_format_mm(bbox_height_mm)}"
                if bbox_width_mm and bbox_height_mm
                else "Unavailable",
            ),
            ("Tumor Pixels", str(white_pixel_count)),
        ],
        "clinical_interpretation": interpretation,
        "doctor_notes": _safe_text(report.get("doctor_notes"), "Pending physician review."),
        "prescription_items": prescription_items,
        "follow_up_items": follow_up_items,
        "disclaimer": (
            "This AI-generated report is a decision-support artifact only. Final diagnosis, treatment, and patient management"
            " must be determined by a qualified clinician after full radiological and clinical review."
        ),
        "images": [
            {"label": "Input Image", "value": input_image, "src": _to_image_src(input_image), "sharpen": False},
            {"label": "Enhanced Image", "value": enhanced_image, "src": _to_image_src(enhanced_image), "sharpen": True},
            {"label": "Boundary Overlay", "value": boundary_image, "src": _to_image_src(boundary_image), "sharpen": True},
        ],
    }


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=PRIMARY_BLUE_DARK,
            alignment=1,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            textColor=MUTED,
            alignment=1,
        ),
        "section": ParagraphStyle(
            "SectionTitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=PRIMARY_BLUE,
            spaceAfter=6,
        ),
        "label": ParagraphStyle(
            "Label",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=MUTED,
        ),
        "value": ParagraphStyle(
            "Value",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=TEXT,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=TEXT,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=PRIMARY_BLUE_DARK,
            alignment=1,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=MUTED,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.8,
            leading=10,
            textColor=MUTED,
        ),
    }


def _build_info_table(rows: List[tuple], styles: dict, col_widths: List[float]) -> Table:
    table_rows = [
        [Paragraph(f"<b>{_safe_text(label)}</b>", styles["body"]), Paragraph(_safe_text(value), styles["body"])]
        for label, value in rows
    ]
    table = Table(table_rows, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), SOFT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _build_image_card(label: str, image_value: Optional[str], styles: dict, width: float, sharpen: bool = False) -> Table:
    image_stream = _prepare_report_image_stream(image_value, sharpen=sharpen)
    if image_stream is None:
        placeholder = Table(
            [[Paragraph("Image unavailable", styles["small"])]],
            colWidths=[width],
            rowHeights=[1.95 * inch],
        )
        placeholder.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
                    ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return Table([[Paragraph(label, styles["caption"])], [placeholder]], colWidths=[width])

    image = Image.open(image_stream)
    ratio = image.size[1] / float(max(image.size[0], 1))
    image_stream.seek(0)
    rendered_image = RLImage(image_stream, width=width, height=min(width * ratio, 2.2 * inch))
    image_table = Table([[rendered_image]], colWidths=[width])
    image_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return Table([[Paragraph(label, styles["caption"])], [image_table]], colWidths=[width])


def build_medical_report_pdf(report: dict) -> io.BytesIO:
    context = build_report_context(report)
    styles = _styles()
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    story = []
    header = Table(
        [
            [
                Paragraph(
                    f"{context['system_name']}<br/><font size='9' color='{MUTED.hexval()}'>{context['system_subtitle']}</font>",
                    styles["body"],
                ),
                Paragraph(context["report_title"], styles["title"]),
                Paragraph(
                    f"<b>Report ID:</b> {context['report_id']}<br/><b>Date:</b> {context['report_date']}",
                    styles["body"],
                ),
            ]
        ],
        colWidths=[55 * mm, 80 * mm, 45 * mm],
    )
    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, -1), 1.0, BORDER),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([header, Spacer(1, 10)])

    patient = context["patient"]
    patient_grid = Table(
        [
            [
                Paragraph(f"<b>Patient Name</b><br/>{patient['name']}", styles["body"]),
                Paragraph(f"<b>Patient ID</b><br/>{patient['patient_id']}", styles["body"]),
            ],
            [
                Paragraph(f"<b>Age</b><br/>{patient['age']}", styles["body"]),
                Paragraph(f"<b>Gender</b><br/>{patient['gender']}", styles["body"]),
            ],
        ],
        colWidths=[88 * mm, 88 * mm],
    )
    patient_grid.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([Paragraph("Patient Information", styles["section"]), patient_grid, Spacer(1, 10)])

    story.extend(
        [
            Paragraph("Scan Details", styles["section"]),
            _build_info_table(context["scan_details"], styles, [58 * mm, 118 * mm]),
            Spacer(1, 10),
            Paragraph("AI Summary", styles["section"]),
            _build_info_table(context["ai_summary"], styles, [58 * mm, 118 * mm]),
            Spacer(1, 10),
            Paragraph("Segmentation Summary", styles["section"]),
            _build_info_table(context["segmentation_summary"], styles, [58 * mm, 118 * mm]),
            Spacer(1, 10),
        ]
    )

    image_cards = [
        _build_image_card(image["label"], image["value"], styles, 80 * mm, sharpen=bool(image.get("sharpen")))
        for image in context["images"]
    ]
    
    # Pack image cards into pairs for the grid
    grid_rows = []
    for i in range(0, len(image_cards), 2):
        row = [image_cards[i]]
        if i + 1 < len(image_cards):
            row.append(image_cards[i+1])
        else:
            row.append("")  # Empty cell
        grid_rows.append(row)
        
    imaging_grid = Table(
        grid_rows,
        colWidths=[88 * mm, 88 * mm],
    )
    imaging_grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([Paragraph("Imaging Results", styles["section"]), imaging_grid, Spacer(1, 10)])

    interpretation_table = Table(
        [[Paragraph(context["clinical_interpretation"], styles["body"])]],
        colWidths=[176 * mm],
    )
    interpretation_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ALERT),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([Paragraph("Clinical Interpretation", styles["section"]), interpretation_table, Spacer(1, 10)])

    doctor_sections = Table(
        [
            [
                Paragraph(f"<b>Doctor Notes</b><br/><br/>{context['doctor_notes']}", styles["body"]),
                Paragraph(
                    "<b>Prescription</b><br/><br/>"
                    + "<br/>".join(f"&#8226; {_safe_text(item)}" for item in context["prescription_items"]),
                    styles["body"],
                ),
                Paragraph(
                    "<b>Follow-up Instructions</b><br/><br/>"
                    + "<br/>".join(f"&#8226; {_safe_text(item)}" for item in context["follow_up_items"]),
                    styles["body"],
                ),
            ]
        ],
        colWidths=[58 * mm, 58 * mm, 60 * mm],
    )
    doctor_sections.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([Paragraph("Clinical Notes", styles["section"]), doctor_sections, Spacer(1, 12)])
    story.append(Paragraph(context["disclaimer"], styles["disclaimer"]))

    document.build(story)
    buffer.seek(0)
    return buffer
