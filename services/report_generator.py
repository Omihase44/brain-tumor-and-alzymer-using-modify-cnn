import base64
import io
import os
from datetime import datetime
from typing import Optional

from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
PRIMARY_BLUE = colors.HexColor("#0B6E99")
PRIMARY_BLUE_DARK = colors.HexColor("#084F6F")
SOFT_BLUE = colors.HexColor("#EAF6FB")
ALERT_SOFT = colors.HexColor("#F6FBFE")
BORDER_COLOR = colors.HexColor("#D5E1E8")
TEXT_COLOR = colors.HexColor("#263744")
MUTED_TEXT = colors.HexColor("#6F7F8C")
SURFACE_COLOR = colors.HexColor("#F7FBFD")


def _safe_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_text(value, default: str = "N/A") -> str:
    if value in (None, ""):
        return default
    return str(value)


def _safe_image_value(value) -> Optional[str]:
    if isinstance(value, dict):
        return value.get("base64") or value.get("image") or value.get("image_base64") or value.get("data") or value.get("path")
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _open_image_stream(image_value: Optional[str]) -> Optional[io.BytesIO]:
    if not image_value:
        return None

    normalized = str(image_value).strip()
    if not normalized:
        return None

    if os.path.exists(normalized):
        try:
            return io.BytesIO(open(normalized, "rb").read())
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

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return buffer


def _resolve_image_dimensions(image_value: Optional[str]) -> Optional[tuple[int, int]]:
    image_stream = _open_image_stream(image_value)
    if image_stream is None:
        return None
    try:
        image = Image.open(image_stream)
        return int(image.size[0]), int(image.size[1])
    except Exception:
        return None


def _parse_percent_number(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        normalized = str(value).strip().replace("%", "")
        if not normalized:
            return None
        numeric = float(normalized)
    except (TypeError, ValueError):
        return None
    if numeric <= 1.0:
        numeric *= 100.0
    return numeric


def _detection_label(report: dict) -> str:
    raw_type = str(report.get("type") or "").strip().lower()
    if raw_type == "brain":
        return "Brain Tumor"
    if raw_type == "alz":
        return "Alzheimer's"
    return raw_type.title() or "Brain Tumor"


def _classify_tumor_size(area_percentage: Optional[float], tumor_area_pixels: int) -> str:
    if tumor_area_pixels <= 0:
        return "No measurable tumor"
    if area_percentage is not None:
        if area_percentage < 1.0:
            return "Small"
        if area_percentage < 3.0:
            return "Medium"
        return "Large"
    if tumor_area_pixels < 1500:
        return "Small"
    if tumor_area_pixels < 6000:
        return "Medium"
    return "Large"


def _confidence_text(confidence_value: object) -> str:
    confidence = _parse_percent_number(confidence_value)
    if confidence is None:
        return "Model confidence could not be quantified from the available output."
    if confidence >= 90.0:
        return "Model confidence indicates high reliability."
    if confidence >= 75.0:
        return "Model confidence indicates good reliability."
    if confidence >= 60.0:
        return "Model confidence indicates moderate reliability."
    return "Model confidence indicates limited reliability and should be reviewed carefully."


def _build_segmentation_context(report: dict) -> dict:
    analysis = _safe_dict(report.get("analysis"))
    tumor = _safe_dict(analysis.get("tumor"))
    segmentation = _safe_dict(analysis.get("segmentation"))
    bounding_box = _safe_dict(segmentation.get("bounding_box"))
    contour_count = int(segmentation.get("contour_count") or 0)

    report_images = _safe_dict(report.get("report_images"))
    original_image_value = (
        _safe_image_value(report.get("input_image"))
        or _safe_image_value(report_images.get("input_image"))
        or _safe_image_value(report.get("image"))
    )
    image_dimensions = _resolve_image_dimensions(original_image_value)
    total_image_area = int(image_dimensions[0] * image_dimensions[1]) if image_dimensions else 0

    tumor_area_pixels = int(segmentation.get("white_pixel_count") or 0)
    if tumor_area_pixels <= 0 and bounding_box:
        tumor_area_pixels = int(bounding_box.get("width") or 0) * int(bounding_box.get("height") or 0)

    tumor_area_percentage = (
        (float(tumor_area_pixels) / float(total_image_area) * 100.0)
        if tumor_area_pixels > 0 and total_image_area > 0
        else None
    )

    detected = bool(tumor.get("detected", report.get("tumor_detected"))) and (
        contour_count > 0 or tumor_area_pixels > 0
    )
    size_label = _classify_tumor_size(tumor_area_percentage, tumor_area_pixels)

    if detected:
        insight = "Localized high-intensity lesion detected in MRI scan."
        interpretation = (
            f"Post-processed segmentation identified a {size_label.lower()} focal lesion"
            f" consistent with {_safe_text(report.get('tumor_type') or tumor.get('tumor_type') or tumor.get('classification'), 'brain lesion')}."
        )
        if tumor_area_percentage is not None:
            interpretation += f" Tumor occupies {tumor_area_percentage:.2f}% of the scan area."
        interpretation += " Clinical and radiological correlation is recommended."
    else:
        insight = "No abnormal tumor region detected."
        interpretation = "No stable focal lesion boundary was retained after segmentation post-processing. Clinical review is still advised if symptoms persist."

    if bounding_box:
        bounding_box_text = (
            f"{int(bounding_box.get('width') or 0)} x {int(bounding_box.get('height') or 0)} px"
        )
    else:
        bounding_box_text = "Unavailable"

    return {
        "detected": detected,
        "contour_count": contour_count,
        "bounding_box": bounding_box,
        "bounding_box_text": bounding_box_text,
        "tumor_area_pixels": tumor_area_pixels,
        "tumor_area_percentage": tumor_area_percentage,
        "size_label": size_label,
        "segmentation_insight": insight,
        "clinical_interpretation": interpretation,
        "confidence_text": _confidence_text(
            report.get("ai_confidence") or report.get("tumor_confidence") or tumor.get("confidence")
        ),
    }


def _collect_ai_result_rows(report: dict, segmentation_context: dict) -> list[list[str]]:
    analysis = _safe_dict(report.get("analysis"))
    tumor = _safe_dict(analysis.get("tumor"))
    area_text = segmentation_context["size_label"]
    if segmentation_context.get("tumor_area_percentage") is not None:
        area_text = f"{area_text} ({segmentation_context['tumor_area_percentage']:.2f}% of scan)"

    detection_status = "Detected" if segmentation_context.get("detected") else "Not Detected"
    rows = [
        ["Detection Status", detection_status],
        ["Detection Type", _detection_label(report)],
        ["Tumor Type", _safe_text(report.get("tumor_type") or tumor.get("tumor_type") or tumor.get("classification"))],
        ["Stage", _safe_text(report.get("tumor_stage") or report.get("tumor_grade") or tumor.get("tumor_stage") or tumor.get("grade"))],
        ["Confidence", _safe_text(report.get("ai_confidence") or report.get("tumor_confidence") or tumor.get("confidence"))],
        ["Model Accuracy", _safe_text(report.get("model_accuracy") or analysis.get("model_accuracy"), "Unavailable")],
        ["Tumor Size", area_text],
        ["Segmentation Region", segmentation_context.get("bounding_box_text") or "Unavailable"],
    ]
    return rows


def _split_bullets(value: Optional[str], fallback: str) -> list[str]:
    text = str(value or fallback).replace("â€¢", "\n").replace("•", "\n").replace(";", "\n")
    items = [item.strip() for item in text.splitlines() if item.strip()]
    return items or [fallback]


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Heading1"], fontName="Times-Bold", fontSize=23, leading=27, textColor=PRIMARY_BLUE_DARK, alignment=1),
        "section": ParagraphStyle("SectionTitle", parent=base["Heading2"], fontName="Times-Bold", fontSize=15, leading=18, textColor=PRIMARY_BLUE, spaceAfter=8),
        "label": ParagraphStyle("MiniLabel", parent=base["BodyText"], fontName="Times-Roman", fontSize=8.5, leading=10, textColor=MUTED_TEXT),
        "value": ParagraphStyle("Value", parent=base["BodyText"], fontName="Times-Bold", fontSize=12.5, leading=15, textColor=TEXT_COLOR),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=14, textColor=TEXT_COLOR),
        "body_bold": ParagraphStyle("BodyBold", parent=base["BodyText"], fontName="Times-Bold", fontSize=10.5, leading=14, textColor=TEXT_COLOR),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Times-Roman", fontSize=9, leading=12, textColor=MUTED_TEXT),
        "small_right": ParagraphStyle("SmallRight", parent=base["BodyText"], fontName="Times-Roman", fontSize=8.5, leading=10, textColor=MUTED_TEXT, alignment=2),
        "value_right": ParagraphStyle("ValueRight", parent=base["BodyText"], fontName="Times-Bold", fontSize=11.5, leading=14, textColor=TEXT_COLOR, alignment=2),
        "summary_value": ParagraphStyle("SummaryValue", parent=base["BodyText"], fontName="Times-Bold", fontSize=13, leading=16, textColor=PRIMARY_BLUE_DARK),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=14, textColor=TEXT_COLOR),
        "disclaimer": ParagraphStyle("Disclaimer", parent=base["Italic"], fontName="Times-Italic", fontSize=8.5, leading=11, textColor=MUTED_TEXT),
        "image_caption": ParagraphStyle("ImageCaption", parent=base["BodyText"], fontName="Times-Bold", fontSize=10, leading=12, textColor=PRIMARY_BLUE_DARK, alignment=1),
        "signature": ParagraphStyle("Signature", parent=base["BodyText"], fontName="Times-Bold", fontSize=10.5, leading=13, textColor=TEXT_COLOR, alignment=1),
    }


def _build_report_image_card(label: str, image_value: Optional[str], styles: dict, width: float, sharpen: bool = False):
    image_stream = _prepare_report_image_stream(image_value, sharpen=sharpen)
    if image_stream is None:
        placeholder = Table([[Paragraph("Image not available", styles["small"])]], colWidths=[width], rowHeights=[2.2 * inch])
        placeholder.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), SURFACE_COLOR),
            ("BOX", (0, 0), (-1, -1), 0.8, BORDER_COLOR),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        return Table([[Paragraph(label, styles["image_caption"])], [placeholder]], colWidths=[width])

    image = Image.open(image_stream)
    ratio = image.size[1] / float(max(image.size[0], 1))
    image_stream.seek(0)
    rendered = RLImage(image_stream, width=width, height=min(width * ratio, 2.6 * inch))
    image_table = Table([[rendered]], colWidths=[width])
    image_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER_COLOR),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return Table([[Paragraph(label, styles["image_caption"])], [image_table]], colWidths=[width])


def build_medical_report_pdf(report: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=34, bottomMargin=30)
    styles = _styles()
    story = []

    analysis = _safe_dict(report.get("analysis"))
    tumor = _safe_dict(analysis.get("tumor"))
    detailed_info = _safe_dict(report.get("detailed_info"))
    segmentation_context = _build_segmentation_context(report)

    report_date = _safe_text(report.get("report_created_at") or report.get("date"), "")
    try:
        if report_date:
            report_date = datetime.fromisoformat(report_date).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        pass

    header = Table(
        [[
            Table([[Paragraph("NeuroDetect AI - Neurodiagnostic Report", styles["body_bold"]), Paragraph("AI-powered Brain Analysis System", styles["small"])]], colWidths=[2.2 * inch]),
            Paragraph("Clinical Diagnostic Summary", styles["title"]),
            Table([[Paragraph("Report ID", styles["small_right"]), Paragraph(_safe_text(report.get("id")), styles["value_right"])], [Paragraph("Date", styles["small_right"]), Paragraph(report_date or "N/A", styles["value_right"])]], colWidths=[0.8 * inch, 1.3 * inch]),
        ]],
        colWidths=[2.3 * inch, 2.4 * inch, 2.1 * inch],
    )
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 1.2, BORDER_COLOR)]))
    story.extend([header, Spacer(1, 14)])

    patient_table = Table(
        [
            [
                Table([[Paragraph("Name", styles["label"]), Paragraph(_safe_text(report.get("patient_name")), styles["value"])]], colWidths=[2.95 * inch]),
                Table([[Paragraph("Age", styles["label"]), Paragraph(_safe_text(report.get("patient_age")), styles["value"])]], colWidths=[2.95 * inch]),
            ],
            [
                Table([[Paragraph("Gender", styles["label"]), Paragraph(_safe_text(report.get("patient_gender")), styles["value"])]], colWidths=[2.95 * inch]),
                Table([[Paragraph("Detection Type", styles["label"]), Paragraph(_detection_label(report), styles["value"])]], colWidths=[2.95 * inch]),
            ],
        ],
        colWidths=[3.05 * inch, 3.05 * inch],
    )
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE_COLOR),
        ("BOX", (0, 0), (-1, -1), 0.9, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, BORDER_COLOR),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([Paragraph("Patient Information", styles["section"]), patient_table, Spacer(1, 14)])

    detection_status = "Detected" if segmentation_context.get("detected") else "Not Detected"
    size_value = segmentation_context["size_label"]
    if segmentation_context.get("tumor_area_percentage") is not None:
        size_value = f"{size_value} ({segmentation_context['tumor_area_percentage']:.2f}% of scan)"
    summary_data = [
        ["Detection Status", detection_status],
        ["Tumor Type", _safe_text(report.get("tumor_type") or tumor.get("tumor_type") or tumor.get("classification"))],
        ["Stage", _safe_text(report.get("tumor_stage") or report.get("tumor_grade") or tumor.get("tumor_stage") or tumor.get("grade"))],
        ["Confidence", _safe_text(report.get("ai_confidence") or report.get("tumor_confidence") or tumor.get("confidence"))],
        ["Tumor Size", size_value],
    ]
    summary_table = Table([[Paragraph(label, styles["label"]), Paragraph(value, styles["summary_value"])] for label, value in summary_data], colWidths=[1.55 * inch, 4.55 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.9, BORDER_COLOR),
        ("LINEBEFORE", (0, 0), (0, -1), 4.0, PRIMARY_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([Paragraph("Summary Section", styles["section"]), summary_table, Spacer(1, 14)])

    insight_table = Table(
        [[Paragraph(f"<b>Segmentation Insight:</b> {segmentation_context['segmentation_insight']}<br/><br/><b>Confidence Interpretation:</b> {segmentation_context['confidence_text']}", styles["callout"])]],
        colWidths=[6.1 * inch],
    )
    insight_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ALERT_SOFT),
        ("BOX", (0, 0), (-1, -1), 0.9, BORDER_COLOR),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([Paragraph("Segmentation Insight", styles["section"]), insight_table, Spacer(1, 14)])

    ai_rows = _collect_ai_result_rows(report, segmentation_context)
    ai_table = Table([[Paragraph(f"<b>{label}</b>", styles["body"]), Paragraph(value, styles["body"])] for label, value in ai_rows], colWidths=[2.1 * inch, 4.0 * inch])
    ai_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF5F9")),
        ("BOX", (0, 0), (-1, -1), 0.9, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([Paragraph("AI Analysis", styles["section"]), ai_table, Spacer(1, 14)])

    report_images = _safe_dict(report.get("report_images"))
    segmentation_payload = _safe_dict(analysis.get("segmentation"))
    input_image = (
        _safe_image_value(report.get("input_image"))
        or _safe_image_value(report_images.get("input_image"))
        or _safe_image_value(report.get("image"))
    )
    enhanced_image = (
        _safe_image_value(report.get("enhanced_image"))
        or _safe_image_value(report.get("enhanced_mri"))
        or _safe_image_value(report_images.get("enhanced_image"))
    )
    boundary_image = (
        _safe_image_value(report.get("segmentation_overlay"))
        or _safe_image_value(report_images.get("segmentation_overlay"))
        or _safe_image_value(report.get("boundary_image"))
        or _safe_image_value(segmentation_payload.get("boundary_image"))
        or _safe_image_value(report.get("segmentation_image"))
        or _safe_image_value(report_images.get("boundary_image"))
    )

    input_card = _build_report_image_card("Input Image", input_image, styles, width=2.8 * inch)
    enhanced_card = _build_report_image_card("Enhanced Image", enhanced_image, styles, width=2.8 * inch)
    boundary_card = _build_report_image_card("Boundary Image", boundary_image, styles, width=5.8 * inch, sharpen=True)
    imaging_grid = Table([[input_card, enhanced_card], [boundary_card, ""]], colWidths=[2.95 * inch, 2.95 * inch])
    imaging_grid.setStyle(TableStyle([
        ("SPAN", (0, 1), (1, 1)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([Paragraph("Imaging Results", styles["section"]), imaging_grid, Spacer(1, 14)])

    clinical_description = _safe_text(
        detailed_info.get("description"),
        "AI-assisted review suggests that the lesion pattern should be correlated with formal radiological assessment.",
    )
    recommended_action = _safe_text(
        detailed_info.get("treatment") or detailed_info.get("recommended_action"),
        "Recommend specialist neurological or neurosurgical evaluation with follow-up imaging as clinically indicated.",
    )
    interpretation_text = (
        f"{segmentation_context['clinical_interpretation']} "
        f"{clinical_description} Recommended action: {recommended_action}"
    )
    interpretation_table = Table([[Paragraph(interpretation_text, styles["callout"])]], colWidths=[6.1 * inch])
    interpretation_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE_COLOR),
        ("BOX", (0, 0), (-1, -1), 0.9, BORDER_COLOR),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([Paragraph("Clinical Interpretation", styles["section"]), interpretation_table, Spacer(1, 14)])

    doctor_notes = _safe_text(report.get("doctor_notes"), "No doctor notes available.")
    prescription_items = _split_bullets(report.get("prescription"), "Supportive medication may be prescribed according to symptoms and clinical findings.")
    follow_up_items = _split_bullets(report.get("follow_up"), "Follow-up specialist review and repeat imaging are advised as clinically indicated.")
    prescription_text = "<br/>".join(f"&#8226; {item}" for item in prescription_items)
    follow_up_text = "<br/>".join(f"&#8226; {item}" for item in follow_up_items)
    doctor_table = Table(
        [[
            Paragraph(f"<b>Doctor Notes</b><br/><br/>{doctor_notes}", styles["body"]),
            Paragraph(f"<b>Prescription</b><br/><br/>{prescription_text}", styles["body"]),
            Paragraph(f"<b>Follow-up Instructions</b><br/><br/>{follow_up_text}", styles["body"]),
        ]],
        colWidths=[2.0 * inch, 2.0 * inch, 2.1 * inch],
    )
    doctor_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.9, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([Paragraph("Doctor Section", styles["section"]), doctor_table, Spacer(1, 18)])

    signature = Table([[Spacer(1, 24)], [Paragraph("<b>Doctor Signature</b>", styles["signature"])], [Paragraph("Neuro Specialist", styles["small"])]], colWidths=[2.2 * inch])
    signature.setStyle(TableStyle([
        ("LINEABOVE", (0, 1), (-1, 1), 0.8, MUTED_TEXT),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    signature_wrap = Table([["", signature]], colWidths=[4.0 * inch, 2.2 * inch])
    signature_wrap.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend([signature_wrap, Spacer(1, 16)])

    story.append(Paragraph("Disclaimer: This AI-assisted report supports clinical review and does not replace independent medical judgment. Final diagnosis and treatment decisions must be confirmed by a qualified healthcare professional after full clinical correlation.", styles["disclaimer"]))

    document.build(story)
    buffer.seek(0)
    return buffer


# Keep legacy imports aligned with the rebuilt reporting stack.
from services.reporting import build_medical_report_pdf, build_report_context  # noqa: E402,F401
