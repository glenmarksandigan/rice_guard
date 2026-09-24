"""
RiceGuard — Flask AI API Server (Unified 2-Step Pipeline)

All models share the same 2-step pipeline:
  Step 1: YOLO rice_leaf_detector.pt  →  Is it a rice leaf?
  Step 2: User's chosen model         →  Classification (healthy or specific disease)

Serves rice disease predictions using 4 models:
  - MobileNetV2 (Keras)
  - DenseNet201 (Keras)
  - EfficientNetV2S (Keras)
  - YOLOv8 Classifier

Returns inference time with every prediction.
Runs on port 5000.
"""

import os
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mysql.connector

from models import predict as model_predict, get_available_models, AVAILABLE_MODELS
from yolo_models import predict_yolo_3step, detect_rice_leaf

app = Flask(__name__)
CORS(app)

# Upload folder for received images
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database config (same as PHP)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'rice_guard_db',
}


def get_db():
    """Get a MySQL database connection."""
    return mysql.connector.connect(**DB_CONFIG)


# ── Routes ─────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    models_list = AVAILABLE_MODELS
    return jsonify({
        'service': 'RiceGuard AI API',
        'version': '4.0.0',
        'status': 'running',
        'pipeline': 'Unified 2-Step (YOLO Gatekeeper → Model Classification)',
        'models': models_list,
        'endpoints': {
            'GET /models': 'List available AI models with accuracy info',
            'GET /pipeline': 'View the unified pipeline diagram',
            'POST /predict': 'Analyze an image (2-step pipeline: YOLO Step 1 → chosen model Step 2)',
            'POST /predict/compare': 'Compare all models on same image (all use YOLO Step 1)',
        }
    })


@app.route('/models', methods=['GET'])
def list_models():
    """Return info on all available AI models."""
    return jsonify(get_available_models())


@app.route('/pipeline', methods=['GET'])
def pipeline_page():
    """Serve the interactive pipeline visualization page."""
    pipeline_html = os.path.join(os.path.dirname(__file__), 'pipeline_diagram.html')
    if os.path.exists(pipeline_html):
        return send_file(pipeline_html)
    return jsonify({'error': 'Pipeline diagram not found'}), 404


@app.route('/pipeline/info', methods=['GET'])
def pipeline_info():
    """Return the pipeline structure as JSON."""
    return jsonify({
        'pipeline_name': 'RiceGuard Unified 2-Step Pipeline',
        'version': '4.0.0',
        'description': 'All models share a single YOLO-based gatekeeper for rice leaf detection, then classify health status or specific disease in one step.',
        'steps': [
            {
                'step': 1,
                'name': 'Rice Leaf Detection',
                'model': 'rice_leaf_detector.pt (YOLOv8)',
                'purpose': 'Detect if the uploaded image contains a rice leaf',
                'outputs': ['is_rice_leaf (bool)', 'confidence (%)', 'cropped_leaf_image'],
                'on_fail': 'Image rejected — not a rice leaf',
                'shared_by': ['mobilenetv2', 'densenet201', 'efficientnetv2s', 'yolov8'],
            },
            {
                'step': 2,
                'name': 'Classification',
                'purpose': 'Classify the rice leaf as healthy or identify the specific disease',
                'models_available': {
                    'mobilenetv2': 'MobileNetV2 (Keras) — Fast and lightweight',
                    'densenet201': 'DenseNet201 (Keras) — Deep and accurate',
                    'efficientnetv2s': 'EfficientNetV2S (Keras) — Best balance',
                    'yolov8': 'YOLOv8 Classifier — rice_disease_classifier.pt',
                },
                'outputs': ['disease_name', 'disease_id', 'confidence_score', 'is_healthy (bool)', 'top_predictions'],
                'depends_on': 'Step 1 must pass',
            },
        ],
        'diseases_detected': [
            'Bacterial Leaf Blight',
            'Leaf Blast',
            'Sheath Blight',
            'Healthy Rice Leaf',
        ],
    })


@app.route('/predict/yolo', methods=['POST'])
def predict_yolo():
    """
    Run 2-Step YOLO Pipeline:
    Step 1: Is it a rice leaf or not?
    Step 2: Classify — healthy or which disease?
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    image_data = image_file.read()
    if len(image_data) == 0:
        return jsonify({'error': 'Image file is empty'}), 400

    result = predict_yolo_3step(image_data)

    # Save to database if report_id provided and step1 passed
    report_id = request.form.get('report_id')
    if report_id and result.get('step1_is_rice_leaf') and result.get('step2_classification'):
        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                """INSERT INTO tbl_ai_diagnosis
                   (report_id, disease_id, confidence_score, model_version)
                   VALUES (%s, %s, %s, %s)""",
                (int(report_id), result['step2_classification']['disease_id'],
                 result['step2_classification']['confidence_score'], result['model_version'])
            )
            db.commit()
            result['ai_diagnosis_id'] = cursor.lastrowid
            cursor.close()
            db.close()
        except Exception as e:
            result['db_warning'] = f'Failed to save diagnosis: {str(e)}'

    return jsonify(result), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Analyze an uploaded rice leaf image for disease.
    UNIFIED 2-STEP PIPELINE: All models use YOLO rice_leaf_detector.pt as Step 1 gatekeeper.
    Step 1: YOLO detects rice leaf → Step 2: Chosen model classifies (healthy or disease).
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    model_name = request.form.get('model', '').lower()
    if not model_name:
        return jsonify({'error': f'Model parameter is required. Available: {AVAILABLE_MODELS + ["yolo"]}'}), 400

    image_data = image_file.read()
    if len(image_data) == 0:
        return jsonify({'error': 'Image file is empty'}), 400

    # Save image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{timestamp}_{image_file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, 'wb') as f:
        f.write(image_data)

    # Handle YOLO model route
    if model_name in ['yolo', 'yolov8']:
        result = predict_yolo_3step(image_data)
        result['model_used'] = 'yolov8'
        result['image_saved'] = filename

        if result.get('step2_classification'):
            result['disease_name'] = result['step2_classification']['disease_name']
            result['disease_id'] = result['step2_classification']['disease_id']
            result['confidence_score'] = result['step2_classification']['confidence_score']
            result['is_healthy'] = result['step2_classification']['is_healthy']

        report_id = request.form.get('report_id')
        if report_id and result.get('disease_id') and result.get('step1_is_rice_leaf'):
            try:
                db = get_db()
                cursor = db.cursor()
                cursor.execute(
                    """INSERT INTO tbl_ai_diagnosis
                       (report_id, disease_id, confidence_score, model_version)
                       VALUES (%s, %s, %s, %s)""",
                    (int(report_id), result['disease_id'],
                     result['confidence_score'], result['model_version'])
                )
                db.commit()
                result['ai_diagnosis_id'] = cursor.lastrowid
                cursor.close()
                db.close()
            except Exception as e:
                result['db_warning'] = f'Failed to save diagnosis: {str(e)}'

        return jsonify(result), 200

    # ══════════════════════════════════════════════════════════════════
    #  UNIFIED 2-STEP PIPELINE for Standard Models (MobileNetV2, DenseNet201, EfficientNetV2S)
    #  Step 1: YOLO rice_leaf_detector.pt (shared gatekeeper)
    #  Step 2: Keras classifier (user's chosen model) — classifies healthy or disease
    # ══════════════════════════════════════════════════════════════════

    import time
    pipeline_start = time.time()

    # ── STEP 1: YOLO Rice Leaf Detection ──
    step1 = detect_rice_leaf(image_data)

    if not step1['is_rice_leaf']:
        return jsonify({
            'status': 'rejected',
            'step1_is_rice_leaf': False,
            'step1_confidence': step1['confidence'],
            'step1_detection_method': step1['detection_method'],
            'step2_classification': None,
            'disease_name': 'No Rice Leaf Detected',
            'message': 'No rice leaf detected in the image. Please upload a clear photo of a rice leaf.',
            'model_version': model_name,
            'image_saved': filename,
            'detector_model_loaded': step1['detector_model_loaded'],
            'pipeline': 'unified_2step',
        }), 200

    # ── STEP 2: Run Keras model on the (possibly cropped) leaf image ──
    # Use cropped image from YOLO detection for better accuracy
    classification_input = step1['cropped_image_bytes'] or image_data

    try:
        result = model_predict(model_name, classification_input)
        is_healthy = (result.get('disease_name') == 'Healthy Rice Leaf')
        result['status'] = 'success'
        result['step1_is_rice_leaf'] = True
        result['step1_confidence'] = step1['confidence']
        result['step1_detection_method'] = step1['detection_method']
        result['is_healthy'] = is_healthy
        result['step2_classification'] = {
            'disease_name': result.get('disease_name'),
            'disease_id': result.get('disease_id'),
            'confidence_score': result.get('confidence_score'),
            'is_healthy': is_healthy,
        }
        result['detector_model_loaded'] = step1['detector_model_loaded']
        result['pipeline'] = 'unified_2step'
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Save to database if report_id provided
    report_id = request.form.get('report_id')
    if report_id:
        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                """INSERT INTO tbl_ai_diagnosis
                   (report_id, disease_id, confidence_score, model_version)
                   VALUES (%s, %s, %s, %s)""",
                (int(report_id), result['disease_id'],
                 result['confidence_score'], result['model_version'])
            )
            db.commit()
            result['ai_diagnosis_id'] = cursor.lastrowid
            cursor.close()
            db.close()
        except Exception as e:
            result['db_warning'] = f'Failed to save diagnosis: {str(e)}'

    result['model_used'] = model_name
    result['image_saved'] = filename

    return jsonify(result), 200


@app.route('/predict/compare', methods=['POST'])
def compare_models():
    """
    Run all models on the same image and return comparative results.
    UNIFIED 2-STEP PIPELINE: YOLO Step 1 runs ONCE, then all classifiers use the result.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_data = image_file.read()

    if len(image_data) == 0:
        return jsonify({'error': 'Image file is empty'}), 400

    import time

    # ── SHARED STEP 1: YOLO Rice Leaf Detection (runs ONCE for all models) ──
    step1 = detect_rice_leaf(image_data)

    if not step1['is_rice_leaf']:
        return jsonify({
            'status': 'rejected',
            'step1_is_rice_leaf': False,
            'step1_confidence': step1['confidence'],
            'step1_detection_method': step1['detection_method'],
            'message': 'No rice leaf detected in the image. Please upload a clear photo of a rice leaf.',
            'predictions': [],
            'detector_model_loaded': step1['detector_model_loaded'],
            'pipeline': 'unified_2step',
        }), 200

    # Use the cropped leaf image for all classifiers
    classification_input = step1['cropped_image_bytes'] or image_data

    results = []
    total_time_ms = 0

    # ── Run all Keras models on the same cropped leaf image ──
    for model_key in AVAILABLE_MODELS:
        prediction = model_predict(model_key, classification_input)
        prediction['model_used'] = model_key
        prediction['step1_is_rice_leaf'] = True
        prediction['step1_confidence'] = step1['confidence']
        prediction['step1_detection_method'] = step1['detection_method']
        prediction['is_healthy'] = (prediction.get('disease_name') == 'Healthy Rice Leaf')
        prediction['detector_model_loaded'] = step1['detector_model_loaded']
        prediction['pipeline'] = 'unified_2step'
        total_time_ms += prediction.get('inference_time_ms', 0)
        results.append(prediction)

    # ── Add YOLO 2-step prediction to comparison ──
    yolo_pred = predict_yolo_3step(image_data)
    if yolo_pred.get('step2_classification'):
        yolo_formatted = {
            'model_used': 'yolov8',
            'disease_name': yolo_pred['step2_classification']['disease_name'],
            'disease_id': yolo_pred['step2_classification']['disease_id'],
            'confidence_score': yolo_pred['step2_classification']['confidence_score'],
            'is_healthy': yolo_pred['step2_classification']['is_healthy'],
            'inference_time_ms': yolo_pred.get('inference_time_ms', 0),
            'model_version': yolo_pred.get('model_version', 'yolov8-2step'),
            'step1_is_rice_leaf': yolo_pred.get('step1_is_rice_leaf'),
            'step1_confidence': yolo_pred.get('step1_confidence'),
            'step1_detection_method': yolo_pred.get('step1_detection_method'),
            'detector_model_loaded': yolo_pred.get('detector_model_loaded'),
            'pipeline': 'unified_2step',
        }
        total_time_ms += yolo_formatted['inference_time_ms']
        results.append(yolo_formatted)

    # Sort by confidence
    results.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)

    return jsonify({
        'pipeline': 'unified_2step',
        'step1_is_rice_leaf': True,
        'step1_confidence': step1['confidence'],
        'step1_detection_method': step1['detection_method'],
        'predictions': results,
        'best_prediction': results[0],
        'total_inference_time_ms': round(total_time_ms, 2),
        'detector_model_loaded': step1['detector_model_loaded'],
    }), 200


# ── Run Server ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  RiceGuard AI API Server v4.0 (Unified 2-Step Pipeline)")
    print("  Running on http://localhost:5000")
    print(f"  Models: {', '.join(AVAILABLE_MODELS + ['yolov8'])}")
    print("  Pipeline: Step 1 YOLO Gatekeeper → Step 2 Classification")
    print("  Endpoints: /predict, /predict/yolo, /pipeline, /models")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
