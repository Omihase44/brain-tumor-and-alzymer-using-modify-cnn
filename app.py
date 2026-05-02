from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file, send_from_directory
from flask_cors import CORS
from functools import wraps
import logging
import os
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import json
import hashlib
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image as PILImage

from routes.analysis_routes import analysis_bp, analyze_medical_image
from routes.appointment_routes import create_appointment_blueprint, register_appointment_socketio
from routes.chat_routes import create_chat_blueprint, register_chat_socketio
from services.data_store import bootstrap_platform_data, init_platform_database, sync_reports_to_db, sync_users_to_db
from services.model_registry import get_model_registry
from services.reporting import build_medical_report_pdf, build_report_context
from socket_handler import emit_report_status_updated, emit_scan_uploaded

try:
    from flask_socketio import SocketIO
except ImportError:  # pragma: no cover - startup fallback until dependency is installed
    class SocketIO:  # type: ignore
        def __init__(self, app=None, **kwargs):
            self.app = app

        def on(self, event_name):
            def decorator(func):
                return func
            return decorator

        def emit(self, *args, **kwargs):
            return None

        def run(self, app, *args, **kwargs):
            return app.run(*args, **kwargs)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOGGER = logging.getLogger("neurodetect")
logging.getLogger('werkzeug').setLevel(logging.ERROR)


def _resolve_runtime_path(env_key, default_name):
    configured_value = os.environ.get(env_key)
    if configured_value:
        return configured_value if os.path.isabs(configured_value) else os.path.join(BASE_DIR, configured_value)
    return os.path.join(BASE_DIR, default_name)


def _ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _ensure_json_file(path, default_value):
    _ensure_parent_dir(path)
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as file_handle:
            json.dump(default_value, file_handle, indent=2)


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-only-secret-key')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

UPLOAD_FOLDER = _resolve_runtime_path("UPLOAD_FOLDER", "uploads")
CHAT_MEDIA_FOLDER = os.path.join(UPLOAD_FOLDER, "chat_media")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'dcm', 'dicom'}
USERS_FILE = _resolve_runtime_path("USERS_FILE", "users.json")
REPORTS_FILE = _resolve_runtime_path("REPORTS_FILE", "reports.json")
REPORTS_DB_FILE = _resolve_runtime_path("REPORTS_DB_FILE", "reports_db.json")
REPORTS_OUTPUT_DIR = _resolve_runtime_path("REPORTS_OUTPUT_DIR", "reports")
PATIENT_DETAILS_FILE = _resolve_runtime_path("PATIENT_DETAILS_FILE", "patient_details.json")
CHAT_DB_PATH = _resolve_runtime_path("CHAT_DB_PATH", "chat_store.sqlite3")
APPOINTMENT_DB_PATH = _resolve_runtime_path("APPOINTMENT_DB_PATH", "appointment_store.sqlite3")
PLATFORM_DB_PATH = _resolve_runtime_path("PLATFORM_DB_PATH", "platform_store_runtime.sqlite3")
DATA_SEED_FILES = (
    (USERS_FILE, {"doctors": [], "patients": []}),
    (REPORTS_FILE, []),
    (REPORTS_DB_FILE, []),
    (PATIENT_DETAILS_FILE, {}),
)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['CHAT_DB_PATH'] = CHAT_DB_PATH
app.config['APPOINTMENT_DB_PATH'] = APPOINTMENT_DB_PATH
app.config['PLATFORM_DB_PATH'] = PLATFORM_DB_PATH
app.config['REPORTS_DB_FILE'] = REPORTS_DB_FILE
app.config['REPORTS_OUTPUT_DIR'] = REPORTS_OUTPUT_DIR

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHAT_MEDIA_FOLDER, exist_ok=True)
os.makedirs(REPORTS_OUTPUT_DIR, exist_ok=True)
for seed_path, default_value in DATA_SEED_FILES:
    _ensure_json_file(seed_path, default_value)
init_platform_database(PLATFORM_DB_PATH)

# Global variables for models
brain_model = None
alz_model = None

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def usernames_match(left, right):
    return normalize_text(left) == normalize_text(right)


def normalize_detection_type(value):
    normalized = normalize_text(value).lower()
    if normalized in {'alz', 'alzheimer', 'alzheimers', "alzheimer's"}:
        return 'alz'
    if normalized in {'combined', 'both', 'all'}:
        return 'combined'
    return 'brain'


def format_confidence_percentage(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith('%'):
            return normalized
        try:
            value = float(normalized)
        except ValueError:
            return normalized

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if numeric_value <= 1:
        numeric_value *= 100
    formatted = f"{numeric_value:.2f}".rstrip('0').rstrip('.')
    return f"{formatted}%"


def _first_non_empty(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _extract_image_payload(value):
    if isinstance(value, dict):
        return (
            value.get('base64')
            or value.get('image')
            or value.get('image_base64')
            or value.get('data')
        )
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def safe_dict(obj):
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


def normalize_metric_payload(value):
    metrics_payload = _coerce_dict(value)
    if not metrics_payload:
        return {}

    if 'f1_score' not in metrics_payload and metrics_payload.get('f1') is not None:
        metrics_payload['f1_score'] = metrics_payload.get('f1')
    for metric_key in ('accuracy', 'precision', 'recall', 'f1_score'):
        normalized_label = format_confidence_percentage(metrics_payload.get(metric_key))
        if normalized_label:
            metrics_payload[f'{metric_key}_label'] = normalized_label
    return metrics_payload


def _coerce_dict(value):
    return safe_dict(value)


def _coerce_list(value):
    return value if isinstance(value, list) else []


def _coerce_dict_list(value):
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _get_request_json_dict(error_message='Invalid JSON format'):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raw_data = request.get_data(cache=True, as_text=True)
        data = safe_dict(raw_data)
    if not isinstance(data, dict) or not data:
        raise ValueError(error_message)
    return data


def normalize_analysis_payload(analysis):
    analysis = _coerce_dict(analysis)
    if not analysis:
        return {}

    tumor_payload = _coerce_dict(analysis.get('tumor'))
    if tumor_payload:
        tumor_payload['confidence'] = format_confidence_percentage(tumor_payload.get('confidence'))
        tumor_payload['type_confidence'] = format_confidence_percentage(tumor_payload.get('type_confidence'))
        tumor_payload['stage_confidence'] = format_confidence_percentage(tumor_payload.get('stage_confidence'))
        tumor_payload['tumor_type'] = tumor_payload.get('tumor_type') or tumor_payload.get('classification')
        tumor_payload['tumor_stage'] = tumor_payload.get('tumor_stage') or tumor_payload.get('grade')
        tumor_payload['model_metrics'] = normalize_metric_payload(tumor_payload.get('model_metrics'))
        if 'tumor_volume_mm3' in tumor_payload and 'volume_mm3' not in tumor_payload:
            tumor_payload['volume_mm3'] = tumor_payload.get('tumor_volume_mm3')
    analysis['tumor'] = tumor_payload

    alzheimers_payload = _coerce_dict(analysis.get('alzheimers'))
    if alzheimers_payload:
        alzheimers_payload['confidence'] = format_confidence_percentage(alzheimers_payload.get('confidence'))
        alzheimers_payload['model_metrics'] = normalize_metric_payload(alzheimers_payload.get('model_metrics'))
        alzheimers_payload['stage'] = _first_non_empty(
            alzheimers_payload.get('stage'),
            analysis.get('alzheimer_stage'),
            analysis.get('alz_stage'),
        )
    analysis['alzheimers'] = alzheimers_payload

    primary_result = _coerce_dict(analysis.get('primary_result'))
    if primary_result and 'confidence' in primary_result:
        primary_result['confidence'] = format_confidence_percentage(primary_result.get('confidence'))
    analysis['primary_result'] = primary_result
    analysis['model_metrics'] = {
        'tumor': normalize_metric_payload(_coerce_dict((_coerce_dict(analysis.get('model_metrics'))).get('tumor'))),
        'alzheimers': normalize_metric_payload(_coerce_dict((_coerce_dict(analysis.get('model_metrics'))).get('alzheimers'))),
    }
    analysis['model_accuracy'] = format_confidence_percentage(analysis.get('model_accuracy'))

    analysis['tumor_confidence'] = format_confidence_percentage(analysis.get('tumor_confidence'))
    analysis['alzheimer_confidence'] = format_confidence_percentage(analysis.get('alzheimer_confidence'))
    analysis['alzheimer_stage'] = _first_non_empty(
        analysis.get('alzheimer_stage'),
        analysis.get('alz_stage'),
        alzheimers_payload.get('stage'),
    )
    analysis['confidence'] = format_confidence_percentage(analysis.get('confidence'))
    return analysis


def resolve_accuracy_label(value=None, metrics=None):
    label = format_confidence_percentage(value)
    if label not in (None, "", "Unavailable"):
        return label

    metrics_payload = normalize_metric_payload(_coerce_dict(metrics))
    return _first_non_empty(
        metrics_payload.get('accuracy_label'),
        format_confidence_percentage(metrics_payload.get('accuracy')),
        label,
        'Unavailable',
    )


def build_filtered_ai_result_payload(analysis, detection_type=None):
    analysis = normalize_analysis_payload(analysis)
    tumor_payload = _coerce_dict(analysis.get('tumor'))
    alzheimers_payload = _coerce_dict(analysis.get('alzheimers'))
    normalized_type = normalize_text(detection_type or analysis.get('analysis_type')).lower()

    tumor_result = {
        'type': 'Brain Tumor',
        'result': 'Detected' if tumor_payload.get('detected') else 'Not Detected',
        'tumor_type': tumor_payload.get('tumor_type') or tumor_payload.get('classification') or 'No Tumor',
        'tumor_stage': tumor_payload.get('tumor_stage') or tumor_payload.get('grade') or 'N/A',
        'confidence': format_confidence_percentage(tumor_payload.get('confidence')) or 'N/A',
        'model_accuracy': resolve_accuracy_label(
            analysis.get('model_accuracy'),
            tumor_payload.get('model_metrics'),
        ),
    }
    alzheimer_result = {
        'type': 'Alzheimer',
        'stage': alzheimers_payload.get('stage') or analysis.get('alzheimer_stage') or 'NonDemented',
        'confidence': format_confidence_percentage(alzheimers_payload.get('confidence')) or 'N/A',
        'model_accuracy': resolve_accuracy_label(
            _first_non_empty(
                normalize_metric_payload(_coerce_dict(alzheimers_payload.get('model_metrics'))).get('accuracy_label'),
                analysis.get('model_accuracy'),
            ),
            alzheimers_payload.get('model_metrics'),
        ),
    }

    if normalized_type in {'brain', 'tumor', 'brain tumor'}:
        return tumor_result
    if normalized_type in {'alz', 'alzheimer', 'alzheimers', "alzheimer's"}:
        return alzheimer_result
    return {
        'brain_tumor': tumor_result,
        'alzheimer': alzheimer_result,
    }


def build_public_analysis_payload(report):
    report = normalize_report_record(report)
    analysis = _coerce_dict(report.get('analysis'))
    tumor_payload = _coerce_dict(analysis.get('tumor'))
    alzheimers_payload = _coerce_dict(analysis.get('alzheimers'))
    segmentation_payload = _coerce_dict(analysis.get('segmentation'))
    insights_payload = _coerce_dict(report.get('ai_clinical_insights')) or _coerce_dict(analysis.get('ai_clinical_insights'))
    contour_count = int(segmentation_payload.get('contour_count') or 0)

    return {
        'tumor': {
            'detected': bool(tumor_payload.get('detected', report.get('tumor_detected'))),
            'classification': tumor_payload.get('classification') or report.get('tumor_type') or report.get('result'),
            'tumor_type': tumor_payload.get('tumor_type') or report.get('tumor_type'),
            'grade': tumor_payload.get('grade') or report.get('tumor_stage'),
            'tumor_stage': tumor_payload.get('tumor_stage') or report.get('tumor_stage'),
            'confidence': format_confidence_percentage(tumor_payload.get('confidence')) or report.get('tumor_confidence'),
        },
        'alzheimers': {
            'detected': bool(alzheimers_payload.get('detected', report.get('alzheimer_detected'))),
            'stage': alzheimers_payload.get('stage') or report.get('alzheimer_stage'),
            'confidence': format_confidence_percentage(alzheimers_payload.get('confidence')) or report.get('alzheimer_confidence'),
        },
        'segmentation': {
            'available': bool(report.get('boundary_image') or report.get('mask_image')),
            'backend': segmentation_payload.get('backend') or 'opencv_contour',
            'contour_count': contour_count,
            'bounding_box': segmentation_payload.get('bounding_box'),
            'white_pixel_count': segmentation_payload.get('white_pixel_count'),
            'tumor_area_percentage': segmentation_payload.get('tumor_area_percentage'),
            'tumor_area_mm2': segmentation_payload.get('tumor_area_mm2'),
            'mask_quality': segmentation_payload.get('mask_quality'),
            'mask_image': _first_non_empty(
                segmentation_payload.get('mask_image'),
                report.get('mask_image'),
                _coerce_dict(report.get('report_images')).get('mask_image'),
            ),
            'boundary_image': _first_non_empty(
                segmentation_payload.get('boundary_image'),
                report.get('boundary_image'),
                _coerce_dict(report.get('report_images')).get('boundary_image'),
            ),
        },
        'ai_clinical_insights': {
            'tumor_type': report.get('tumor_type'),
            'tumor_stage': report.get('tumor_stage'),
            'tumor_confidence': report.get('tumor_confidence'),
            'alzheimer_stage': report.get('alzheimer_stage'),
            'model_accuracy': report.get('model_accuracy'),
            'boundary_status': _first_non_empty(
                insights_payload.get('boundary_status'),
                insights_payload.get('segmentation_boundary'),
                'Detected' if contour_count else 'Unavailable',
            ),
        },
        'model_accuracy': report.get('model_accuracy'),
        'confidence': report.get('ai_confidence'),
    }


def build_public_report_payload(report):
    report = normalize_report_record(report)
    report_images = _coerce_dict(report.get('report_images'))
    public_analysis = build_public_analysis_payload(report)
    public_report_images = {
        'input_image': _first_non_empty(report_images.get('input_image'), report.get('input_image')),
        'enhanced_image': _first_non_empty(report_images.get('enhanced_image'), report.get('enhanced_image')),
        'mask_image': _first_non_empty(report_images.get('mask_image'), report.get('mask_image')),
        'boundary_image': _first_non_empty(report_images.get('boundary_image'), report.get('boundary_image')),
    }

    return {
        'id': report.get('id'),
        'patient_id': report.get('patient_id'),
        'patient_name': report.get('patient_name'),
        'patient_age': report.get('patient_age'),
        'patient_gender': report.get('patient_gender'),
        'type': report.get('type'),
        'result': report.get('result'),
        'symptoms': report.get('symptoms'),
        'notes': report.get('notes'),
        'detailed_info': _coerce_dict(report.get('detailed_info')),
        'date': report.get('date'),
        'status': report.get('status'),
        'doctor_notes': report.get('doctor_notes'),
        'prescription': report.get('prescription'),
        'follow_up': report.get('follow_up'),
        'reviewed_by': report.get('reviewed_by'),
        'reviewed_date': report.get('reviewed_date'),
        'approved_by': report.get('approved_by'),
        'approved_date': report.get('approved_date'),
        'rejected_by': report.get('rejected_by'),
        'rejected_date': report.get('rejected_date'),
        'sent_date': report.get('sent_date'),
        'ai_confidence': report.get('ai_confidence'),
        'tumor_confidence': report.get('tumor_confidence'),
        'tumor_type': report.get('tumor_type'),
        'tumor_stage': report.get('tumor_stage'),
        'tumor_detected': report.get('tumor_detected'),
        'alzheimer_confidence': report.get('alzheimer_confidence'),
        'alzheimer_stage': report.get('alzheimer_stage'),
        'alzheimer_detected': report.get('alzheimer_detected'),
        'model_accuracy': report.get('model_accuracy'),
        'ai_clinical_insights': _coerce_dict(public_analysis.get('ai_clinical_insights')),
        'analysis': public_analysis,
        'ai_results': report.get('ai_results'),
        'report_images': public_report_images,
        'input_image': public_report_images.get('input_image'),
        'enhanced_image': public_report_images.get('enhanced_image'),
        'mask_image': public_report_images.get('mask_image'),
        'boundary_image': public_report_images.get('boundary_image'),
        'download_enabled': bool(report.get('download_enabled')),
        'report_ready': bool(report.get('report_ready')),
        'report_created_at': report.get('report_created_at'),
        'report_download_url': resolve_report_download_url(report.get('id')) if report.get('report_ready') else None,
        'report_preview_url': f"/report-preview/{int(report.get('id'))}" if report.get('id') not in (None, '') else None,
    }


def build_prediction_response_payload(analysis, detection_type, input_image=None):
    analysis = normalize_analysis_payload(analysis)
    analysis_images = _coerce_dict(analysis.get('images'))
    filtered_prediction = build_filtered_ai_result_payload(analysis, detection_type)
    confidence = _first_non_empty(
        _coerce_dict(filtered_prediction).get('confidence'),
        _coerce_dict(_coerce_dict(filtered_prediction).get('brain_tumor')).get('confidence'),
        _coerce_dict(_coerce_dict(filtered_prediction).get('alzheimer')).get('confidence'),
        analysis.get('confidence'),
    )
    images = {
        'input_image': _first_non_empty(
            input_image,
            _extract_image_payload(analysis_images.get('input')),
            _extract_image_payload(analysis_images.get('original')),
            analysis.get('input_image_base64'),
            analysis.get('original_image_base64'),
        ),
        'enhanced_image': _first_non_empty(
            _extract_image_payload(analysis_images.get('enhanced')),
            analysis.get('enhanced_image_base64'),
        ),
        'mask_image': _first_non_empty(
            _extract_image_payload(analysis_images.get('mask')),
            analysis.get('mask_image_base64'),
        ),
        'boundary_image': _first_non_empty(
            _extract_image_payload(analysis_images.get('boundary')),
            _extract_image_payload(analysis_images.get('overlay')),
            analysis.get('boundary_image_base64'),
            analysis.get('segmentation_image'),
        ),
    }

    return {
        'prediction': filtered_prediction,
        'confidence': confidence,
        'images': images,
        'input_image': images.get('input_image'),
        'enhanced_image': images.get('enhanced_image'),
        'mask_image': images.get('mask_image'),
        'boundary_image': images.get('boundary_image'),
        'ai_result': filtered_prediction,
    }


def _log_single_ai_result(result_payload):
    result_payload = _coerce_dict(result_payload)
    result_type = result_payload.get('type')
    if result_type == 'Brain Tumor':
        LOGGER.info('Prediction: Brain Tumor')
        LOGGER.info('Confidence: %s', result_payload.get('confidence') or 'N/A')
        LOGGER.info('Tumor Type: %s', result_payload.get('tumor_type') or 'N/A')
        LOGGER.info('Tumor Stage/Grade: %s', result_payload.get('tumor_stage') or 'N/A')
        LOGGER.info('Model Accuracy: %s', result_payload.get('model_accuracy') or 'Unavailable')
        return
    if result_type == 'Alzheimer':
        LOGGER.info('Prediction: Alzheimer')
        LOGGER.info('Confidence: %s', result_payload.get('confidence') or 'N/A')
        LOGGER.info('Stage: %s', result_payload.get('stage') or 'N/A')
        LOGGER.info('Model Accuracy: %s', result_payload.get('model_accuracy') or 'Unavailable')


def log_prediction_summary(analysis, detection_type):
    LOGGER.info('Model Loaded')
    filtered_result = build_filtered_ai_result_payload(analysis, detection_type)
    if isinstance(filtered_result, dict) and filtered_result.get('type'):
        _log_single_ai_result(filtered_result)
        return
    for key in ('brain_tumor', 'alzheimer'):
        _log_single_ai_result(_coerce_dict(_coerce_dict(filtered_result).get(key)))


def load_reports_db():
    if os.path.exists(REPORTS_DB_FILE):
        with open(REPORTS_DB_FILE, 'r', encoding='utf-8') as file_handle:
            return _coerce_dict_list(json.load(file_handle))
    return []


def save_reports_db(report_entries):
    with open(REPORTS_DB_FILE, 'w', encoding='utf-8') as file_handle:
        json.dump(_coerce_dict_list(report_entries), file_handle, indent=2)


def get_report_registry_entry(report_id):
    if report_id in (None, ''):
        return {}
    for entry in load_reports_db():
        try:
            if int(entry.get('report_id', -1)) == int(report_id):
                return entry
        except (TypeError, ValueError):
            continue
    return {}


def upsert_report_registry_entry(metadata):
    metadata = _coerce_dict(metadata)
    if not metadata:
        return {}

    existing_entries = load_reports_db()
    updated_entries = []
    replaced = False
    for entry in existing_entries:
        try:
            same_report = int(entry.get('report_id', -1)) == int(metadata.get('report_id', -2))
        except (TypeError, ValueError):
            same_report = False
        if same_report:
            updated_entries.append({**entry, **metadata})
            replaced = True
        else:
            updated_entries.append(entry)
    if not replaced:
        updated_entries.append(metadata)
    save_reports_db(updated_entries)
    return metadata


def delete_report_registry_entry(report_id):
    filtered_entries = []
    for entry in load_reports_db():
        try:
            same_report = int(entry.get('report_id', -1)) == int(report_id)
        except (TypeError, ValueError):
            same_report = False
        if not same_report:
            filtered_entries.append(entry)
    save_reports_db(filtered_entries)


def resolve_saved_report_path(report_id):
    return os.path.join(REPORTS_OUTPUT_DIR, f'{int(report_id)}.pdf')


def resolve_report_download_url(report_id):
    return f'/generate-report/{int(report_id)}'


def persist_report_pdf(report):
    normalized_report = normalize_report_record(report)
    report_id = int(normalized_report.get('id'))
    output_path = resolve_saved_report_path(report_id)
    report_buffer = build_medical_report_pdf(normalized_report)
    with open(output_path, 'wb') as file_handle:
        file_handle.write(report_buffer.getvalue())

    created_at = datetime.now().isoformat()
    metadata = {
        'report_id': report_id,
        'patient_id': normalized_report.get('patient_id'),
        'report_ready': True,
        'created_at': created_at,
        'pdf_path': output_path,
    }
    upsert_report_registry_entry(metadata)
    normalized_report['report_ready'] = True
    normalized_report['download_enabled'] = True
    normalized_report['report_file'] = output_path
    normalized_report['report_created_at'] = created_at
    LOGGER.info('Report Generated: %s', os.path.basename(output_path))
    return normalized_report, metadata


def get_accessible_saved_report(report_id, patient_id=None):
    metadata = get_report_registry_entry(report_id)
    if not metadata:
        return {}, None

    resolved_path = metadata.get('pdf_path') or resolve_saved_report_path(report_id)
    if patient_id is not None:
        try:
            if int(metadata.get('patient_id', -1)) != int(patient_id):
                return {}, None
        except (TypeError, ValueError):
            return {}, None
    if not os.path.exists(resolved_path):
        return metadata, None
    return metadata, resolved_path


def build_report_image_bundle(report):
    analysis = _coerce_dict(report.get('analysis'))
    analysis = normalize_analysis_payload(analysis) if analysis else {}
    analysis_images = _coerce_dict(analysis.get('images'))
    segmentation = _coerce_dict(analysis.get('segmentation'))
    enhancement = _coerce_dict(analysis.get('enhancement'))

    existing_images = _coerce_dict(report.get('report_images'))
    normalized_images = {
        'input_image': _first_non_empty(
            existing_images.get('input_image'),
            report.get('input_image'),
            report.get('image'),
            report.get('original_mri'),
            report.get('original_image'),
            _extract_image_payload(analysis_images.get('input')),
            _extract_image_payload(analysis_images.get('original')),
            analysis.get('input_image_base64'),
            analysis.get('original_image_base64'),
            enhancement.get('input_image_base64'),
            enhancement.get('original_image_base64'),
        ),
        'enhanced_image': _first_non_empty(
            existing_images.get('enhanced_image'),
            existing_images.get('enhanced_mri'),
            report.get('enhanced_mri'),
            report.get('enhanced_image'),
            _extract_image_payload(analysis_images.get('enhanced')),
            analysis.get('enhanced_image_base64'),
            enhancement.get('enhanced_image_base64'),
        ),
        'mask_image': _first_non_empty(
            existing_images.get('mask_image'),
            report.get('mask_image'),
            report.get('segmentation_mask'),
            _extract_image_payload(analysis_images.get('mask')),
            _extract_image_payload(segmentation.get('mask_image')),
            _extract_image_payload(segmentation.get('segmentation_mask')),
            analysis.get('mask_image_base64'),
        ),
        'boundary_image': _first_non_empty(
            existing_images.get('boundary_image'),
            existing_images.get('segmentation_overlay'),
            report.get('boundary_image'),
            report.get('segmentation_overlay'),
            report.get('segmentation_image'),
            report.get('segmented_image'),
            _extract_image_payload(analysis_images.get('boundary')),
            _extract_image_payload(analysis_images.get('overlay')),
            _extract_image_payload(segmentation.get('boundary_image')),
            _extract_image_payload(segmentation.get('overlay_image')),
            analysis.get('boundary_image_base64'),
            analysis.get('segmentation_image'),
        ),
    }
    return normalized_images


def normalize_report_record(report):
    if not isinstance(report, dict):
        return report

    report['analysis'] = normalize_analysis_payload(report.get('analysis'))
    report_images = build_report_image_bundle(report)
    report['report_images'] = report_images
    report_metadata = get_report_registry_entry(report.get('id'))

    report['image'] = _first_non_empty(report.get('image'), report_images.get('input_image'))
    report['input_image'] = _first_non_empty(report.get('input_image'), report_images.get('input_image'), report.get('image'))
    report['original_mri'] = _first_non_empty(report.get('original_mri'), report_images.get('input_image'))
    report['original_image'] = _first_non_empty(report.get('original_image'), report['original_mri'], report['input_image'])
    report['enhanced_image'] = _first_non_empty(report.get('enhanced_image'), report_images.get('enhanced_image'))
    report['enhanced_mri'] = _first_non_empty(report.get('enhanced_mri'), report['enhanced_image'])
    report['mask_image'] = _first_non_empty(report.get('mask_image'), report_images.get('mask_image'))
    report['segmentation_mask'] = _first_non_empty(report.get('segmentation_mask'), report['mask_image'])
    report['boundary_image'] = _first_non_empty(report.get('boundary_image'), report_images.get('boundary_image'))
    report['segmentation_overlay'] = _first_non_empty(report.get('segmentation_overlay'), report['boundary_image'])
    report['segmentation_image'] = _first_non_empty(report.get('segmentation_image'), report['boundary_image'])
    report['segmented_image'] = _first_non_empty(report.get('segmented_image'), report['boundary_image'])
    report['report_ready'] = bool(
        report_metadata.get('report_ready')
        or report.get('report_ready')
        or report.get('download_enabled')
    )
    report['download_enabled'] = bool(report['report_ready'])
    report['report_file'] = _first_non_empty(report.get('report_file'), report_metadata.get('pdf_path'))
    report['report_created_at'] = _first_non_empty(report.get('report_created_at'), report_metadata.get('created_at'))
    report['tumor_type'] = _first_non_empty(
        report.get('tumor_type'),
        (report['analysis'].get('tumor') or {}).get('tumor_type'),
        (report['analysis'].get('tumor') or {}).get('classification'),
        report.get('result'),
    )
    report['tumor_stage'] = _first_non_empty(
        report.get('tumor_stage'),
        (report['analysis'].get('tumor') or {}).get('tumor_stage'),
        (report['analysis'].get('tumor') or {}).get('grade'),
        report.get('tumor_grade'),
    )
    report['alzheimer_stage'] = _first_non_empty(
        report.get('alzheimer_stage'),
        (report['analysis'].get('alzheimers') or {}).get('stage'),
        report['analysis'].get('alzheimer_stage'),
        report.get('alz_stage'),
    )

    tumor_confidence = format_confidence_percentage(
        _first_non_empty(
            report.get('tumor_confidence'),
            (report['analysis'].get('tumor') or {}).get('confidence'),
            _coerce_dict(report.get('model_confidences')).get('tumor'),
        )
    )
    alzheimer_confidence = format_confidence_percentage(
        _first_non_empty(
            report.get('alzheimer_confidence'),
            (report['analysis'].get('alzheimers') or {}).get('confidence'),
            _coerce_dict(report.get('model_confidences')).get('alzheimers'),
        )
    )
    ai_confidence = format_confidence_percentage(
        _first_non_empty(
            report.get('ai_confidence'),
            report.get('confidence'),
            (report['analysis'].get('primary_result') or {}).get('confidence'),
            report['analysis'].get('confidence'),
            tumor_confidence,
            alzheimer_confidence,
        )
    )

    report['tumor_confidence'] = tumor_confidence
    report['alzheimer_confidence'] = alzheimer_confidence
    report['ai_confidence'] = ai_confidence
    report['confidence'] = ai_confidence
    report['model_metrics'] = _coerce_dict(report.get('model_metrics')) or _coerce_dict(report['analysis'].get('model_metrics'))
    if isinstance(report['model_metrics'], dict):
        report['model_metrics']['tumor'] = normalize_metric_payload(_coerce_dict(report['model_metrics'].get('tumor')))
        report['model_metrics']['alzheimers'] = normalize_metric_payload(_coerce_dict(report['model_metrics'].get('alzheimers')))
    tumor_accuracy = normalize_metric_payload(_coerce_dict(report['model_metrics'].get('tumor'))).get('accuracy_label') if isinstance(report['model_metrics'], dict) else None
    alzheimer_accuracy = normalize_metric_payload(_coerce_dict(report['model_metrics'].get('alzheimers'))).get('accuracy_label') if isinstance(report['model_metrics'], dict) else None
    report['model_accuracy'] = _first_non_empty(
        format_confidence_percentage(report.get('model_accuracy')),
        tumor_accuracy,
        alzheimer_accuracy,
        normalize_metric_payload(_coerce_dict((report['analysis'].get('tumor') or {}).get('model_metrics'))).get('accuracy_label'),
        normalize_metric_payload(_coerce_dict((report['analysis'].get('alzheimers') or {}).get('model_metrics'))).get('accuracy_label'),
    )
    report['ai_clinical_insights'] = _coerce_dict(report.get('ai_clinical_insights')) or _coerce_dict(report['analysis'].get('ai_clinical_insights'))
    if isinstance(report['ai_clinical_insights'], dict):
        report['ai_clinical_insights']['alzheimer_stage'] = _first_non_empty(
            report['ai_clinical_insights'].get('alzheimer_stage'),
            report['alzheimer_stage'],
        )
        report['ai_clinical_insights']['model_accuracy'] = _first_non_empty(
            report['ai_clinical_insights'].get('model_accuracy'),
            report['model_accuracy'],
        )
        report['ai_clinical_insights']['boundary_status'] = _first_non_empty(
            report['ai_clinical_insights'].get('boundary_status'),
            report['ai_clinical_insights'].get('segmentation_boundary'),
            'Detected' if _coerce_dict(report['analysis'].get('segmentation')).get('contour_count') else 'Unavailable',
        )

    model_confidences = _coerce_dict(report.get('model_confidences'))
    if model_confidences:
        model_confidences['tumor'] = tumor_confidence
        model_confidences['alzheimers'] = alzheimer_confidence
        report['model_confidences'] = model_confidences

    analysis_tumor = report['analysis'].get('tumor')
    if isinstance(analysis_tumor, dict):
        analysis_tumor['confidence'] = tumor_confidence
        analysis_tumor['tumor_type'] = report['tumor_type']
        analysis_tumor['tumor_stage'] = report['tumor_stage']
        analysis_tumor['model_metrics'] = normalize_metric_payload(
            _coerce_dict(analysis_tumor.get('model_metrics')) or _coerce_dict(report['model_metrics'].get('tumor'))
        )
    analysis_alzheimers = report['analysis'].get('alzheimers')
    if isinstance(analysis_alzheimers, dict):
        analysis_alzheimers['confidence'] = alzheimer_confidence
        analysis_alzheimers['stage'] = report['alzheimer_stage']
        analysis_alzheimers['model_metrics'] = normalize_metric_payload(
            _coerce_dict(analysis_alzheimers.get('model_metrics')) or _coerce_dict(report['model_metrics'].get('alzheimers'))
        )
    if isinstance(report['analysis'].get('primary_result'), dict) and report['analysis']['primary_result'].get('confidence') is None:
        report['analysis']['primary_result']['confidence'] = ai_confidence
    report['analysis']['confidence'] = ai_confidence
    report['analysis']['model_accuracy'] = report['model_accuracy']
    report['analysis']['alzheimer_stage'] = report['alzheimer_stage']
    analysis_images = _coerce_dict(report['analysis'].get('images'))
    report['analysis']['images'] = analysis_images
    analysis_segmentation = _coerce_dict(report['analysis'].get('segmentation'))
    if report.get('mask_image') is not None:
        analysis_segmentation['mask_image'] = report.get('mask_image')
    if report.get('boundary_image') is not None:
        analysis_segmentation['boundary_image'] = report.get('boundary_image')
    report['analysis']['segmentation'] = analysis_segmentation

    asset_paths = _coerce_dict(report.get('asset_paths'))
    if report.get('mask_image') and not asset_paths.get('mask_image'):
        asset_paths['mask_image'] = report.get('mask_image_path')
    if report.get('boundary_image') and not asset_paths.get('boundary_image'):
        asset_paths['boundary_image'] = report.get('overlay_image_path') or report.get('boundary_image_path')
    report['asset_paths'] = asset_paths
    report['report_images'] = {
        'input_image': report.get('input_image'),
        'enhanced_image': report.get('enhanced_image'),
        'mask_image': report.get('mask_image'),
        'boundary_image': report.get('boundary_image'),
    }
    report['ai_results'] = build_filtered_ai_result_payload(report['analysis'], report.get('type'))

    return report

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            loaded = safe_dict(json.load(f))
            users_payload = {
                "doctors": _coerce_dict_list(loaded.get("doctors")),
                "patients": _coerce_dict_list(loaded.get("patients")),
            }
            sync_users_to_db(app.config['PLATFORM_DB_PATH'], users_payload)
            return users_payload
    users_payload = {"doctors": [], "patients": []}
    sync_users_to_db(app.config['PLATFORM_DB_PATH'], users_payload)
    return users_payload

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)
    sync_users_to_db(app.config['PLATFORM_DB_PATH'], users)

def load_reports():
    reports_file_path = REPORTS_FILE
    if os.path.exists(reports_file_path):
        file_size = os.path.getsize(reports_file_path)
        # Skip loading if file is larger than 100MB to prevent memory issues
        if file_size > 100 * 1024 * 1024:  # 100MB
            print(f"[WARNING] Reports file is too large ({file_size / (1024*1024):.1f}MB), skipping bootstrap. Reports will be loaded on-demand.")
            sync_reports_to_db(app.config['PLATFORM_DB_PATH'], [])
            return []
        
        with open(reports_file_path, 'r') as f:
            reports = [normalize_report_record(report) for report in _coerce_dict_list(json.load(f))]
            sync_reports_to_db(app.config['PLATFORM_DB_PATH'], reports)
            return reports
    sync_reports_to_db(app.config['PLATFORM_DB_PATH'], [])
    return []

def save_reports(reports):
    with open(REPORTS_FILE, 'w') as f:
        json.dump([normalize_report_record(report) for report in reports], f, indent=2)
    sync_reports_to_db(app.config['PLATFORM_DB_PATH'], reports)

def load_patient_details():
    if os.path.exists(PATIENT_DETAILS_FILE):
        with open(PATIENT_DETAILS_FILE, 'r') as f:
            return _coerce_dict(json.load(f))
    return {}

def save_patient_details(details):
    with open(PATIENT_DETAILS_FILE, 'w') as f:
        json.dump(details, f, indent=2)


def build_ai_assist_payload(report):
    report = normalize_report_record(report)
    analysis_payload = _coerce_dict(report.get('analysis'))
    tumor_payload = _coerce_dict(analysis_payload.get('tumor'))
    alzheimer_payload = _coerce_dict(analysis_payload.get('alzheimers'))
    segmentation_payload = _coerce_dict(analysis_payload.get('segmentation'))
    normalized_type = normalize_detection_type(report.get('type'))
    tumor_detected = bool(tumor_payload.get('detected', report.get('tumor_detected')))
    alzheimer_detected = bool(alzheimer_payload.get('detected', report.get('alzheimer_detected')))
    tumor_type = report.get('tumor_type') or tumor_payload.get('tumor_type') or tumor_payload.get('classification') or 'brain lesion'
    tumor_stage = report.get('tumor_stage') or tumor_payload.get('tumor_stage') or tumor_payload.get('grade') or 'N/A'
    tumor_confidence = report.get('tumor_confidence') or tumor_payload.get('confidence') or 'N/A'
    alzheimer_stage = report.get('alzheimer_stage') or alzheimer_payload.get('stage') or 'NonDemented'
    alzheimer_confidence = report.get('alzheimer_confidence') or alzheimer_payload.get('confidence') or 'N/A'
    contour_count = int(segmentation_payload.get('contour_count') or 0)
    boundary_phrase = (
        f'{contour_count} red boundary contour{"s" if contour_count != 1 else ""} outlined on the original MRI'
        if contour_count
        else 'no reliable focal red boundary outlined on the MRI'
    )

    if normalized_type == 'brain':
        if tumor_detected:
            return {
                'doctor_notes': (
                    f'MRI analysis suggests presence of {tumor_type} with confidence {tumor_confidence} '
                    f'and estimated stage {tumor_stage}. The image processing pipeline identified {boundary_phrase}. '
                    'Clinical correlation with symptoms and specialist review is recommended.'
                ),
                'prescription': (
                    'Recommend corticosteroids to reduce edema if clinically indicated, together with symptom-guided '
                    'supportive care and further contrast-enhanced neuroimaging review.'
                ),
                'follow_up': (
                    'Follow-up MRI in 4-6 weeks. Neurosurgery or neurology consultation is advised, '
                    'with earlier review if symptoms worsen.'
                ),
            }
        return {
            'doctor_notes': (
                f'MRI analysis does not show a convincing focal tumor pattern. Current classifier confidence is '
                f'{tumor_confidence}, and the contour pipeline found {boundary_phrase}. Continued clinical monitoring is recommended.'
            ),
            'prescription': (
                'Provide symptomatic care as needed and continue routine neurological observation according to the clinical presentation.'
            ),
            'follow_up': (
                'Repeat imaging if symptoms persist or worsen. Clinical follow-up with the treating physician is advised.'
            ),
        }

    if normalized_type == 'alz':
        return {
            'doctor_notes': (
                f'MRI analysis suggests Alzheimer stage {alzheimer_stage} with confidence {alzheimer_confidence}. '
                'Clinical cognitive assessment and neurological correlation are recommended.'
            ),
            'prescription': (
                'Recommend cognitive evaluation, medication review, and supportive neurocognitive management per specialist guidance.'
            ),
            'follow_up': (
                'Neurology consultation is advised with repeat cognitive and imaging review in 4-6 weeks or as clinically indicated.'
            ),
        }

    note_parts = []
    if tumor_detected:
        note_parts.append(
            f'Brain MRI suggests {tumor_type} with confidence {tumor_confidence} and estimated stage {tumor_stage}; {boundary_phrase}.'
        )
    if alzheimer_detected:
        note_parts.append(
            f'Concurrent Alzheimer pattern is estimated at stage {alzheimer_stage} with confidence {alzheimer_confidence}.'
        )
    if not note_parts:
        note_parts.append(
            'Combined AI analysis does not show a convincing focal tumor pattern or clear Alzheimer-stage abnormality on the current scan.'
        )
    return {
        'doctor_notes': ' '.join(note_parts) + ' Clinical correlation and multidisciplinary review are recommended.',
        'prescription': (
            'Recommend symptom-guided supportive care, targeted neurological evaluation, and additional imaging or cognitive testing as clinically indicated.'
        ),
        'follow_up': (
            'Arrange specialist follow-up within 4-6 weeks, with earlier reassessment if neurological or cognitive symptoms progress.'
        ),
    }


def finalize_report_submission(report, submission_data, status, submitted_by, timestamp_field):
    report['doctor_notes'] = submission_data.get('doctor_notes', '')
    report['prescription'] = submission_data.get('prescription', '')
    report['follow_up'] = submission_data.get('follow_up', '')
    report['status'] = status
    report['report_ready'] = True
    report['download_enabled'] = True
    report['submitted_by'] = submitted_by
    report['submitted_date'] = datetime.now().isoformat()
    report[timestamp_field] = report['submitted_date']

    normalized_report, metadata = persist_report_pdf(report)
    report.update(normalized_report)
    return metadata


def get_latest_ready_report_entry(patient_id):
    ready_reports = []
    for entry in load_reports_db():
        try:
            same_patient = int(entry.get('patient_id', -1)) == int(patient_id)
        except (TypeError, ValueError):
            same_patient = False
        if same_patient and entry.get('report_ready'):
            ready_reports.append(entry)
    ready_reports.sort(key=lambda item: item.get('created_at', ''), reverse=True)
    return ready_reports[0] if ready_reports else {}

# Authentication decorator
def login_required(user_type=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': 'Please login first'}), 401
            if user_type and session.get('user_type') != user_type:
                return jsonify({'error': 'Unauthorized access'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def load_brain_model():
    global brain_model
    if brain_model is None:
        brain_model = get_model_registry().get_tumor_model().model
        if brain_model is None:
            LOGGER.warning("Brain model not found. Using fallback mode.")
        else:
            LOGGER.info("Model Loaded: Brain classifier")
    return brain_model

def load_alz_model():
    global alz_model
    if alz_model is None:
        alz_model = get_model_registry().get_alzheimer_model().model
        if alz_model is None:
            LOGGER.warning("Alzheimer model not found. Using fallback mode.")
        else:
            LOGGER.info("Model Loaded: Alzheimer classifier")
    return alz_model

def get_detailed_data(cls):
    data = {
        "glioma tumor": {
            "symptoms": "Severe headaches, nausea, vomiting, seizures, cognitive changes, vision problems, balance issues",
            "treatment": "Surgery, radiation therapy, chemotherapy, targeted therapy, regular MRI monitoring",
            "description": "Glioma tumors originate in the glial cells of the brain. They can be aggressive and require immediate medical attention.",
            "risk_factors": "Family history, age (45-65), radiation exposure",
            "prevention": "Regular check-ups, healthy lifestyle, avoid radiation exposure",
            "severity": "High",
            "urgency": "Immediate medical attention required"
        },
        "meningioma tumor": {
            "symptoms": "Headaches, hearing loss, vision problems, memory loss, seizures, weakness in limbs",
            "treatment": "Surgical removal, radiation therapy, observation for small tumors, regular follow-ups",
            "description": "Meningiomas are typically benign tumors that grow from the meninges, the protective layers around the brain.",
            "risk_factors": "Age (60+), female gender, genetic disorders, radiation exposure",
            "prevention": "Regular neurological check-ups, manage risk factors",
            "severity": "Medium",
            "urgency": "Consult neurologist soon"
        },
        "no tumor": {
            "symptoms": "No tumor-related symptoms detected",
            "treatment": "No treatment required. Regular health check-ups recommended",
            "description": "No abnormal growth detected in the brain scan. Continue maintaining a healthy lifestyle.",
            "risk_factors": "N/A",
            "prevention": "Maintain healthy diet, regular exercise, avoid smoking, limit alcohol",
            "severity": "None",
            "urgency": "No immediate action needed"
        },
        "pituitary tumor": {
            "symptoms": "Vision problems, headaches, hormonal imbalances, weight changes, fatigue, infertility",
            "treatment": "Surgery, radiation therapy, medication to control hormone levels, regular monitoring",
            "description": "Pituitary tumors develop in the pituitary gland and can affect hormone production throughout the body.",
            "risk_factors": "Genetic conditions, family history, age (30-40)",
            "prevention": "Regular endocrine check-ups, healthy lifestyle",
            "severity": "Medium-High",
            "urgency": "Schedule appointment with endocrinologist"
        },
        "MildDementia": {
            "symptoms": "Forgetfulness, difficulty finding words, losing things, trouble with planning, mood changes",
            "treatment": "Cognitive therapy, medication (cholinesterase inhibitors), lifestyle modifications, support groups",
            "description": "Early stage of dementia with noticeable cognitive decline that affects daily activities.",
            "risk_factors": "Age (65+), family history, cardiovascular disease, diabetes",
            "prevention": "Mental exercises, physical activity, healthy diet, social engagement",
            "severity": "Mild",
            "urgency": "Early intervention recommended"
        },
        "ModerateDementia": {
            "symptoms": "Significant memory loss, confusion, personality changes, difficulty with daily tasks, sleep disturbances",
            "treatment": "Medications, structured routine, caregiver support, safety modifications, occupational therapy",
            "description": "Moderate stage dementia with clear cognitive impairment requiring assistance with daily activities.",
            "risk_factors": "Age, genetics, previous head injuries, lifestyle factors",
            "prevention": "Early intervention, manage cardiovascular health, brain exercises",
            "severity": "Moderate",
            "urgency": "Medical care required"
        },
        "NonDementia": {
            "symptoms": "Normal cognitive function, no signs of dementia detected",
            "treatment": "No treatment required. Preventive measures recommended",
            "description": "No dementia indicators detected. Brain appears to be functioning normally.",
            "risk_factors": "N/A",
            "prevention": "Brain-healthy lifestyle: exercise, mental stimulation, social connection, Mediterranean diet",
            "severity": "None",
            "urgency": "No immediate action needed"
        },
        "VeryMildDementia": {
            "symptoms": "Very mild cognitive changes, occasional forgetfulness, minimal impact on daily life",
            "treatment": "Monitoring, cognitive exercises, lifestyle changes, regular follow-ups",
            "description": "Very early stage with subtle cognitive changes that may not significantly affect daily functioning.",
            "risk_factors": "Age, family history, cardiovascular health",
            "prevention": "Brain training, physical exercise, stress management, healthy diet",
            "severity": "Very Mild",
            "urgency": "Preventive measures recommended"
        },
        "MCI": {
            "symptoms": "Mild memory changes, subtle lapses in concentration, occasional difficulty with complex tasks",
            "treatment": "Monitoring, lifestyle changes, cognitive stimulation, regular neurological follow-up",
            "description": "Mild Cognitive Impairment is an early clinical stage where cognitive changes are present but daily independence is usually preserved.",
            "risk_factors": "Age, family history, cardiovascular disease, diabetes, prior brain injury",
            "prevention": "Regular exercise, sleep hygiene, cognitive activity, blood pressure and glucose control",
            "severity": "Very Mild",
            "urgency": "Early specialist review recommended"
        },
        "Early": {
            "symptoms": "Persistent forgetfulness, word-finding difficulty, reduced task planning and mild confusion",
            "treatment": "Cognitive therapy, medication review, memory support planning, regular clinical assessment",
            "description": "Early Alzheimer staging suggests clinically noticeable decline that benefits from early intervention and structured care planning.",
            "risk_factors": "Age, genetics, cardiovascular disease, sedentary lifestyle",
            "prevention": "Cognitive exercise, Mediterranean diet, physical activity, vascular risk control",
            "severity": "Mild",
            "urgency": "Neurology consultation advised"
        },
        "Moderate": {
            "symptoms": "Increasing confusion, functional decline, impaired daily activities, mood and behavior changes",
            "treatment": "Medication optimization, caregiver planning, occupational therapy, safety modifications",
            "description": "Moderate Alzheimer staging indicates meaningful cognitive and functional impairment requiring structured support.",
            "risk_factors": "Age, genetics, chronic disease burden, prior neurodegeneration",
            "prevention": "Early treatment adherence, structured routines, cardiovascular health management",
            "severity": "Moderate",
            "urgency": "Ongoing medical supervision required"
        },
        "Severe": {
            "symptoms": "Profound memory loss, communication difficulty, dependence for daily living, behavioral and mobility changes",
            "treatment": "Comprehensive dementia care, caregiver support, fall prevention, nutrition and safety management",
            "description": "Severe Alzheimer staging reflects advanced neurocognitive decline and high clinical care needs.",
            "risk_factors": "Progressive neurodegenerative disease",
            "prevention": "Early-stage management may slow progression, but advanced disease needs supportive care planning",
            "severity": "High",
            "urgency": "Specialist-led dementia care required"
        }
    }
    alias_map = {
        "Early Stage": "Early",
        "Moderate Stage": "Moderate",
        "Severe Stage": "Severe",
    }
    normalized_cls = alias_map.get(cls, cls)
    return data.get(normalized_cls, {
        "symptoms": "Unknown",
        "treatment": "Consult a specialist",
        "description": "Please consult with a healthcare professional for accurate diagnosis",
        "risk_factors": "Please consult a doctor",
        "prevention": "Regular health check-ups recommended",
        "severity": "Unknown",
        "urgency": "Consult doctor"
    })


bootstrap_platform_data(app.config['PLATFORM_DB_PATH'], load_users(), load_reports())

app.register_blueprint(analysis_bp)
app.register_blueprint(
    create_chat_blueprint(
        load_users,
        login_required,
        app.config['CHAT_DB_PATH'],
        save_users,
        app.config['UPLOAD_FOLDER'],
    )
)
app.register_blueprint(
    create_appointment_blueprint(
        login_required,
        load_users,
        save_users,
        app.config['APPOINTMENT_DB_PATH']
    )
)
register_chat_socketio(socketio, load_users, save_users, app.config['CHAT_DB_PATH'])
register_appointment_socketio(socketio, app.config['APPOINTMENT_DB_PATH'])

# Routes
@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/healthz')
def healthcheck():
    return jsonify({'status': 'ok'}), 200

@app.route('/patient')
def patient_portal():
    return render_template('patient.html')

@app.route('/doctor')
def doctor_portal():
    return render_template('doctor.html')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(uploaded_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    image_bytes = uploaded_file.read()
    if not image_bytes:
        return jsonify({'error': 'Uploaded image is empty'}), 400

    detection_type = normalize_detection_type(
        request.form.get('type') or request.form.get('detection_type') or 'brain'
    )

    try:
        analysis = analyze_medical_image(
            image_bytes=image_bytes,
            detection_type=detection_type,
            voxel_metadata={
                'pixel_spacing_x': request.form.get('pixel_spacing_x'),
                'pixel_spacing_y': request.form.get('pixel_spacing_y'),
                'slice_thickness': request.form.get('slice_thickness'),
            },
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    log_prediction_summary(analysis, detection_type)
    input_image = base64.b64encode(image_bytes).decode('utf-8')
    return jsonify({
        'success': True,
        **build_prediction_response_payload(analysis, detection_type, input_image=input_image),
    })


@app.route('/doctor-submit', methods=['POST'])
@login_required('doctor')
def doctor_submit():
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    report_id = data.get('report_id')
    if report_id in (None, ''):
        return jsonify({'error': 'report_id is required'}), 400

    reports = load_reports()
    report = next((item for item in reports if int(item.get('id', -1)) == int(report_id)), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404

    report['approved_by'] = session['full_name']
    metadata = finalize_report_submission(
        report,
        data,
        status='approved',
        submitted_by=session['full_name'],
        timestamp_field='approved_date',
    )

    save_reports(reports)
    try:
        emit_report_status_updated(socketio, report, doctor_id=session.get('user_id'))
    except Exception:
        pass

    return jsonify({
        'success': True,
        'download_enabled': True,
        'report_ready': True,
        'report_download_url': resolve_report_download_url(report['id']),
        'report_metadata': metadata,
        'report': build_public_report_payload(report),
    })


@app.route('/generate-report')
@app.route('/generate-report/<int:report_id>')
def generate_report(report_id=None):
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401

    report_id = report_id if report_id is not None else request.args.get('report_id', type=int)
    if report_id is None:
        return jsonify({'error': 'report_id is required'}), 400

    reports = load_reports()
    if session.get('user_type') == 'patient':
        report = next((item for item in reports if item['id'] == report_id and item['patient_id'] == session['user_id']), None)
    else:
        report = next((item for item in reports if item['id'] == report_id), None)

    if not report:
        return jsonify({'error': 'Report not found'}), 404
    if not report.get('report_ready'):
        return jsonify({'error': 'Report download is available after doctor submission'}), 403

    normalized_report, metadata = persist_report_pdf(report)
    report.update(normalized_report)
    save_reports(reports)
    saved_report_path = metadata.get('pdf_path')

    return send_file(saved_report_path, as_attachment=True, download_name=os.path.basename(saved_report_path), mimetype='application/pdf')


@app.route('/report-preview/<int:report_id>')
@login_required()
def preview_report(report_id):
    reports = load_reports()
    if session.get('user_type') == 'patient':
        report = next((item for item in reports if item['id'] == report_id and item['patient_id'] == session['user_id']), None)
    else:
        report = next((item for item in reports if item['id'] == report_id), None)

    if not report:
        return jsonify({'error': 'Report not found'}), 404

    normalized_report = normalize_report_record(report)
    return render_template(
        'report_clinical.html',
        report=normalized_report,
        report_context=build_report_context(normalized_report),
    )


@app.route('/patient-report/<int:patient_id>')
@login_required()
def get_patient_ready_report(patient_id):
    if session.get('user_type') == 'patient' and int(session.get('user_id', -1)) != int(patient_id):
        return jsonify({'error': 'Unauthorized access'}), 403

    report_entry = get_latest_ready_report_entry(patient_id)
    if not report_entry:
        return jsonify({'patient_id': patient_id, 'report_ready': False, 'report_id': None})

    return jsonify({
        'patient_id': patient_id,
        'report_ready': True,
        'report_id': report_entry.get('report_id'),
        'created_at': report_entry.get('created_at'),
    })

@app.route('/api/patient/register', methods=['POST'])
def patient_register():
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    username = normalize_text(data.get('username'))
    email = normalize_text(data.get('email'))
    password = data.get('password')
    full_name = normalize_text(data.get('full_name'))
    age = normalize_text(data.get('age'))
    gender = normalize_text(data.get('gender'))
    phone = normalize_text(data.get('phone'))
    address = normalize_text(data.get('address'))
    medical_history = normalize_text(data.get('medical_history'))
    
    if not all([username, email, password, full_name]):
        return jsonify({'error': 'All fields are required'}), 400
    
    users = load_users()
    
    if any(usernames_match(u.get('username'), username) for u in users['patients']):
        return jsonify({'error': 'Username already exists'}), 400
    
    new_patient = {
        'id': len(users['patients']) + 1,
        'username': username,
        'email': email,
        'password': hash_password(password),
        'full_name': full_name,
        'age': age,
        'gender': gender,
        'phone': phone,
        'address': address,
        'medical_history': medical_history,
        'created_at': datetime.now().isoformat(),
        'assigned_doctor': None
    }
    
    users['patients'].append(new_patient)
    save_users(users)
    
    patient_details = load_patient_details()
    patient_details[str(new_patient['id'])] = {
        'full_name': full_name,
        'age': age,
        'gender': gender,
        'phone': phone,
        'address': address,
        'medical_history': medical_history,
        'reports': []
    }
    save_patient_details(patient_details)
    
    return jsonify({'success': True, 'message': 'Registration successful'})

@app.route('/api/doctor/register', methods=['POST'])
def doctor_register():
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    username = normalize_text(data.get('username'))
    email = normalize_text(data.get('email'))
    password = data.get('password')
    full_name = normalize_text(data.get('full_name'))
    specialization = normalize_text(data.get('specialization'))
    license_number = normalize_text(data.get('license_number'))
    hospital = normalize_text(data.get('hospital'))
    experience = normalize_text(data.get('experience'))
    
    if not all([username, email, password, full_name, specialization, license_number]):
        return jsonify({'error': 'All fields are required'}), 400
    
    users = load_users()
    
    if any(usernames_match(u.get('username'), username) for u in users['doctors']):
        return jsonify({'error': 'Username already exists'}), 400
    
    new_doctor = {
        'id': len(users['doctors']) + 1,
        'username': username,
        'email': email,
        'password': hash_password(password),
        'full_name': full_name,
        'specialization': specialization,
        'license_number': license_number,
        'hospital': hospital,
        'experience': experience,
        'created_at': datetime.now().isoformat(),
        'patients': []
    }
    
    users['doctors'].append(new_doctor)
    save_users(users)
    
    return jsonify({'success': True, 'message': 'Registration successful'})

@app.route('/api/patient/login', methods=['POST'])
def patient_login():
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    username = normalize_text(data.get('username'))
    password = data.get('password')
    
    users = load_users()
    patient = next((p for p in users['patients'] if usernames_match(p.get('username'), username)), None)
    
    if patient and patient['password'] == hash_password(password):
        session['user_id'] = patient['id']
        session['username'] = normalize_text(patient['username'])
        session['user_type'] = 'patient'
        session['full_name'] = patient['full_name']
        
        return jsonify({
            'success': True,
            'user_type': 'patient',
            'username': normalize_text(patient['username']),
            'full_name': patient['full_name'],
            'patient_id': patient['id']
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/doctor/login', methods=['POST'])
def doctor_login():
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    username = normalize_text(data.get('username'))
    password = data.get('password')
    
    users = load_users()
    doctor = next((d for d in users['doctors'] if usernames_match(d.get('username'), username)), None)
    
    if doctor and doctor['password'] == hash_password(password):
        session['user_id'] = doctor['id']
        session['username'] = normalize_text(doctor['username'])
        session['user_type'] = 'doctor'
        session['full_name'] = doctor['full_name']
        
        return jsonify({
            'success': True,
            'user_type': 'doctor',
            'username': normalize_text(doctor['username']),
            'full_name': doctor['full_name'],
            'doctor_id': doctor['id']
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/check-auth')
def check_auth():
    if 'user_id' in session:
        payload = {
            'authenticated': True,
            'user_id': session.get('user_id'),
            'user_type': session.get('user_type'),
            'username': session.get('username'),
            'full_name': session.get('full_name')
        }
        if session.get('user_type') == 'patient':
            payload['patient_id'] = session.get('user_id')
        elif session.get('user_type') == 'doctor':
            payload['doctor_id'] = session.get('user_id')
        return jsonify(payload)
    return jsonify({'authenticated': False})

@app.route('/api/patient/upload', methods=['POST'])
@login_required('patient')
def patient_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    detection_type = normalize_detection_type(request.form.get('type', 'brain'))
    symptoms = request.form.get('symptoms', '')
    notes = request.form.get('notes', '')
    patient_name = request.form.get('patient_name', '')
    patient_age = request.form.get('patient_age', '')
    patient_gender = request.form.get('patient_gender', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            image_bytes = file.read()
            if not image_bytes:
                return jsonify({'error': 'Uploaded image is empty'}), 400

            analysis = analyze_medical_image(
                image_bytes=image_bytes,
                detection_type=detection_type,
                voxel_metadata={
                    'pixel_spacing_x': request.form.get('pixel_spacing_x'),
                    'pixel_spacing_y': request.form.get('pixel_spacing_y'),
                    'slice_thickness': request.form.get('slice_thickness')
                }
            )
            analysis = safe_dict(analysis)
            if not analysis:
                return jsonify({'error': 'Invalid analysis result'}), 500
            log_prediction_summary(analysis, detection_type)

            tumor_result = safe_dict(analysis.get('tumor'))
            alzheimer_result = safe_dict(analysis.get('alzheimers'))
            if not tumor_result or not alzheimer_result:
                return jsonify({'error': 'Invalid model output'}), 500

            if detection_type == 'brain':
                result = tumor_result.get('classification') or ('Tumor Detected' if tumor_result.get('detected') else 'No Tumor')
                confidence = format_confidence_percentage(tumor_result.get('confidence'))
            else:
                result = alzheimer_result.get('stage') if alzheimer_result.get('detected') else 'NonDementia'
                confidence = format_confidence_percentage(alzheimer_result.get('confidence'))

            detailed_info = get_detailed_data(result)
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            prediction_payload = build_prediction_response_payload(analysis, detection_type, input_image=img_base64)
            report_images = {
                'input_image': img_base64,
                'enhanced_image': prediction_payload.get('enhanced_image'),
                'mask_image': prediction_payload.get('mask_image'),
                'boundary_image': prediction_payload.get('boundary_image'),
            }
            
            reports = load_reports()
            report = normalize_report_record({
                'id': len(reports) + 1,
                'patient_id': session['user_id'],
                'patient_name': patient_name or session.get('full_name'),
                'patient_age': patient_age,
                'patient_gender': patient_gender,
                'type': detection_type,
                'result': result,
                'symptoms': symptoms,
                'notes': notes,
                'detailed_info': detailed_info,
                'date': datetime.now().isoformat(),
                'image': img_base64,
                'input_image': img_base64,
                'original_image': analysis.get('input_image_base64') or img_base64,
                'original_mri': analysis.get('input_image_base64') or img_base64,
                'original_image_path': analysis.get('input_image'),
                'enhanced_image': analysis.get('enhanced_image_base64'),
                'enhanced_mri': analysis.get('enhanced_image_base64'),
                'enhanced_image_path': analysis.get('enhanced_image'),
                'mask_image': prediction_payload.get('mask_image'),
                'mask_image_path': analysis.get('mask_image'),
                'segmentation_mask': prediction_payload.get('mask_image'),
                'filename': filename,
                'status': 'pending',
                'report_ready': False,
                'download_enabled': False,
                'doctor_notes': '',
                'prescription': '',
                'follow_up': '',
                'ai_result': result,
                'ai_results': prediction_payload.get('ai_result'),
                'ai_confidence': confidence,
                'tumor_confidence': tumor_result.get('confidence'),
                'tumor_type': tumor_result.get('tumor_type') or tumor_result.get('classification'),
                'alzheimer_confidence': alzheimer_result.get('confidence'),
                'analysis': analysis,
                'tumor_detected': tumor_result.get('detected', False),
                'tumor_grade': tumor_result.get('grade'),
                'tumor_stage': tumor_result.get('tumor_stage') or tumor_result.get('grade'),
                'tumor_volume_mm3': tumor_result.get('volume_mm3'),
                'alzheimer_detected': alzheimer_result.get('detected', False),
                'alzheimer_stage': alzheimer_result.get('stage'),
                'model_accuracy': _first_non_empty(
                    _coerce_dict(prediction_payload.get('ai_result')).get('model_accuracy'),
                    analysis.get('model_accuracy'),
                ),
                'model_metrics': analysis.get('model_metrics'),
                'ai_clinical_insights': analysis.get('ai_clinical_insights'),
                'boundary_image': prediction_payload.get('boundary_image'),
                'segmentation_image': prediction_payload.get('boundary_image'),
                'segmentation_overlay': prediction_payload.get('boundary_image'),
                'segmented_image': prediction_payload.get('boundary_image'),
                'overlay_image_path': analysis.get('boundary_image'),
                'study_id': analysis.get('study_id'),
                'report_images': report_images,
                'asset_paths': {
                    'input_image': None,
                    'original_image': analysis.get('input_image') or analysis.get('original_image'),
                    'enhanced_image': analysis.get('enhanced_image'),
                    'mask_image': analysis.get('mask_image'),
                    'boundary_image': analysis.get('boundary_image'),
                },
                'model_confidences': {
                    'tumor': tumor_result.get('confidence'),
                    'alzheimers': alzheimer_result.get('confidence'),
                }
            })
            
            reports.append(report)
            save_reports(reports)
            
            patient_details = load_patient_details()
            if str(session['user_id']) in patient_details:
                patient_details[str(session['user_id'])]['reports'].append(report['id'])
                save_patient_details(patient_details)

            users_payload = load_users()
            patient_record = next(
                (patient for patient in users_payload.get('patients', []) if int(patient.get('id', -1)) == int(session['user_id'])),
                None,
            )
            assigned_doctor = patient_record.get('assigned_doctor') if isinstance(patient_record, dict) else None
            try:
                emit_scan_uploaded(
                    socketio,
                    report,
                    doctor_id=int(assigned_doctor) if assigned_doctor not in (None, "", 0) else None,
                )
            except Exception:
                pass
            
            return jsonify({
                'success': True,
                'result': result,
                'confidence': confidence,
                'model_accuracy': report.get('model_accuracy'),
                'detailed_info': detailed_info,
                'report_id': report['id'],
                'report_ready': report.get('report_ready', False),
                'download_enabled': report.get('download_enabled', False),
                **prediction_payload,
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/patient/reports')
@login_required('patient')
def get_patient_reports():
    reports = load_reports()
    patient_reports = [r for r in reports if r['patient_id'] == session['user_id']]
    patient_reports.sort(key=lambda x: x['date'], reverse=True)
    return jsonify({'reports': [build_public_report_payload(report) for report in patient_reports]})

@app.route('/api/patient/report/<int:report_id>')
@login_required('patient')
def get_patient_report(report_id):
    reports = load_reports()
    report = next((r for r in reports if r['id'] == report_id and r['patient_id'] == session['user_id']), None)
    if report:
        return jsonify({'report': build_public_report_payload(report)})
    return jsonify({'error': 'Report not found'}), 404

@app.route('/api/patient/report/<int:report_id>/download')
@login_required('patient')
def download_report_pdf(report_id):
    reports = load_reports()
    report = next((r for r in reports if r['id'] == report_id and r['patient_id'] == session['user_id']), None)
    
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    if not report.get('report_ready'):
        return jsonify({'error': 'Report download is available after doctor submission'}), 403

    normalized_report, metadata = persist_report_pdf(report)
    report.update(normalized_report)
    save_reports(reports)
    saved_report_path = metadata.get('pdf_path')
    return send_file(saved_report_path, as_attachment=True, download_name=os.path.basename(saved_report_path), mimetype='application/pdf')

@app.route('/api/doctor/patients')
@login_required('doctor')
def get_doctor_patients():
    reports = load_reports()
    users = load_users()
    
    patient_ids = list(set(r['patient_id'] for r in reports))
    patients = []
    
    for pid in patient_ids:
        patient = next((p for p in users['patients'] if p['id'] == pid), None)
        if patient:
            patient_reports = [r for r in reports if r['patient_id'] == pid]
            pending_reports = [r for r in patient_reports if r.get('status') == 'pending']
            approved_reports = [r for r in patient_reports if r.get('status') in ['approved', 'sent']]
            patients.append({
                'id': patient['id'],
                'name': patient['full_name'],
                'age': patient.get('age', 'N/A'),
                'gender': patient.get('gender', 'N/A'),
                'email': patient.get('email', ''),
                'phone': patient.get('phone', ''),
                'reports_count': len(patient_reports),
                'pending_reports': len(pending_reports),
                'approved_reports': len(approved_reports),
                'last_report': patient_reports[-1]['date'] if patient_reports else None
            })
    
    return jsonify({'patients': patients})

@app.route('/api/doctor/patient/<int:patient_id>')
@login_required('doctor')
def get_patient_details(patient_id):
    users = load_users()
    reports = load_reports()
    patient_details = load_patient_details()
    
    patient = next((p for p in users['patients'] if p['id'] == patient_id), None)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    patient_reports = [r for r in reports if r['patient_id'] == patient_id]
    patient_reports.sort(key=lambda x: x['date'], reverse=True)
    
    details = patient_details.get(str(patient_id), {})
    
    return jsonify({
        'patient': {
            'id': patient['id'],
            'full_name': patient['full_name'],
            'email': patient['email'],
            'age': patient.get('age', 'N/A'),
            'gender': patient.get('gender', 'N/A'),
            'phone': patient.get('phone', ''),
            'address': patient.get('address', ''),
            'medical_history': patient.get('medical_history', ''),
            'created_at': patient.get('created_at', '')
        },
        'reports': [build_public_report_payload(report) for report in patient_reports],
        'additional_info': details
    })

@app.route('/api/doctor/report/<int:report_id>')
@login_required('doctor')
def get_report(report_id):
    reports = load_reports()
    report = next((r for r in reports if r['id'] == report_id), None)
    if report:
        return jsonify({'report': build_public_report_payload(report)})
    return jsonify({'error': 'Report not found'}), 404


@app.route('/api/doctor/report/<int:report_id>/ai-assist')
@login_required('doctor')
def get_doctor_ai_assist(report_id):
    reports = load_reports()
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    LOGGER.info('AI Assist prepared for report %s', report_id)
    return jsonify({'success': True, 'ai_assist': build_ai_assist_payload(report)})

@app.route('/api/doctor/report/<int:report_id>/update', methods=['POST'])
@login_required('doctor')
def update_report(report_id):
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    reports = load_reports()
    
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    report['doctor_notes'] = data.get('doctor_notes', '')
    report['prescription'] = data.get('prescription', '')
    report['follow_up'] = data.get('follow_up', '')
    report['status'] = 'reviewed'
    report['report_ready'] = False
    report['download_enabled'] = False
    report['report_file'] = None
    report['report_created_at'] = None
    report['reviewed_by'] = session['full_name']
    report['reviewed_date'] = datetime.now().isoformat()
    delete_report_registry_entry(report_id)
    
    save_reports(reports)
    try:
        emit_report_status_updated(socketio, report, doctor_id=session.get('user_id'))
    except Exception:
        pass
    
    return jsonify({
        'success': True,
        'message': 'Report saved as draft',
        'download_enabled': False,
        'report_ready': False,
        'report': build_public_report_payload(report),
    })

@app.route('/api/doctor/report/<int:report_id>/approve', methods=['POST'])
@login_required('doctor')
def approve_report(report_id):
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    reports = load_reports()
    
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    report['approved_by'] = session['full_name']
    metadata = finalize_report_submission(
        report,
        data,
        status='approved',
        submitted_by=session['full_name'],
        timestamp_field='approved_date',
    )
    
    save_reports(reports)
    try:
        emit_report_status_updated(socketio, report, doctor_id=session.get('user_id'))
    except Exception:
        pass
    
    return jsonify({
        'success': True,
        'message': 'Report approved',
        'download_enabled': True,
        'report_ready': True,
        'report_download_url': resolve_report_download_url(report_id),
        'report_metadata': metadata,
        'report': build_public_report_payload(report),
    })

@app.route('/api/doctor/report/<int:report_id>/reject', methods=['POST'])
@login_required('doctor')
def reject_report(report_id):
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    reports = load_reports()
    
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    report['doctor_notes'] = data.get('doctor_notes', '')
    report['status'] = 'rejected'
    report['report_ready'] = False
    report['download_enabled'] = False
    report['report_file'] = None
    report['report_created_at'] = None
    report['rejected_by'] = session['full_name']
    report['rejected_date'] = datetime.now().isoformat()
    delete_report_registry_entry(report_id)
    
    save_reports(reports)
    try:
        emit_report_status_updated(socketio, report, doctor_id=session.get('user_id'))
    except Exception:
        pass
    
    return jsonify({
        'success': True,
        'message': 'Report rejected',
        'download_enabled': False,
        'report_ready': False,
        'report': build_public_report_payload(report),
    })

@app.route('/api/doctor/report/<int:report_id>/send', methods=['POST'])
@login_required('doctor')
def send_report_to_patient(report_id):
    try:
        data = _get_request_json_dict()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    reports = load_reports()
    
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    report['reviewed_by'] = session['full_name']
    metadata = finalize_report_submission(
        report,
        data,
        status='sent',
        submitted_by=session['full_name'],
        timestamp_field='sent_date',
    )
    
    save_reports(reports)
    try:
        emit_report_status_updated(socketio, report, doctor_id=session.get('user_id'))
    except Exception:
        pass
    
    return jsonify({
        'success': True,
        'message': 'Report sent to patient',
        'download_enabled': True,
        'report_ready': True,
        'report_download_url': resolve_report_download_url(report_id),
        'report_metadata': metadata,
        'report': build_public_report_payload(report),
    })

@app.route('/api/doctor/stats')
@login_required('doctor')
def get_doctor_stats():
    reports = load_reports()
    
    total_patients = len(set(r['patient_id'] for r in reports))
    pending_reports = len([r for r in reports if r.get('status') == 'pending'])
    approved_reports = len([r for r in reports if r.get('status') == 'approved'])
    sent_reports = len([r for r in reports if r.get('status') == 'sent'])
    
    return jsonify({
        'total_patients': total_patients,
        'pending_reports': pending_reports,
        'approved_reports': approved_reports,
        'sent_reports': sent_reports
    })


@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({
        "success": False,
        "error": str(e)
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    debug_mode = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'}
    print(f"==================================================")
    print(f"🚀 NeuroDetect Server is running at http://localhost:{port}")
    print(f"==================================================")
    socketio.run(app, debug=debug_mode, use_reloader=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
