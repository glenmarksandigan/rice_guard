"""
RiceGuard — AI Disease Detection Models
Compatible with models saved by the original training script that used:
    @tf.keras.utils.register_keras_serializable(package=model_name)
    def apply_preprocess(img): return preprocess_fn(img)
"""

import os, time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras.applications import mobilenet_v2, densenet, efficientnet_v2
    REAL_MODELS_AVAILABLE = True
    print("✅ TensorFlow loaded")
except ImportError:
    REAL_MODELS_AVAILABLE = False
    print("⚠️ TensorFlow not installed — using simulated models")

MODELS_DIR   = os.path.join(os.path.dirname(__file__), 'trained_models')
NUM_CLASSES  = 4
CLASS_LABELS = ['Bacterial Leaf Blight', 'Healthy Rice Leaf', 'Leaf Blast', 'Sheath Blight']
IMG_SIZE     = (224, 224)
AVAILABLE_MODELS = ['mobilenetv2', 'densenet201', 'efficientnetv2s']

if REAL_MODELS_AVAILABLE:

    # ── Register the exact functions the training script serialised ───────────
    # Training used: @register_keras_serializable(package=model_name)
    # which produces keys like 'MobileNetV2>apply_preprocess'.
    # We must register identically-named functions BEFORE load_model() is called.

    @tf.keras.utils.register_keras_serializable(package='MobileNetV2')
    def apply_preprocess(img):
        return tf.keras.applications.mobilenet_v2.preprocess_input(img)

    @tf.keras.utils.register_keras_serializable(package='DenseNet201')
    def apply_preprocess(img):  # noqa: F811
        return tf.keras.applications.densenet.preprocess_input(img)

    @tf.keras.utils.register_keras_serializable(package='EfficientNetV2S')
    def apply_preprocess(img):  # noqa: F811
        return tf.keras.applications.efficientnet_v2.preprocess_input(img)

    _MODEL_FILES = {
        'mobilenetv2':     'MobileNetV2_rice_disease.keras',
        'densenet201':     'DenseNet201_rice_disease.keras',
        'efficientnetv2s': 'EfficientNetV2S_rice_disease.keras',
    }

    _LOADED_MODELS = {}

    def _get_model(model_key: str) -> tf.keras.Model:
        if model_key in _LOADED_MODELS:
            return _LOADED_MODELS[model_key]

        filepath = os.path.join(MODELS_DIR, _MODEL_FILES[model_key])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")

        print(f"🔄 [{model_key}] loading…")
        model = tf.keras.models.load_model(filepath, compile=False)

        # Sanity check — output must sum to ~1.0
        dummy = tf.zeros((1, *IMG_SIZE, 3))
        out   = model(dummy, training=False).numpy()
        print(f"  🧪 Output sum: {out.sum():.6f}  (should be ~1.0)")
        if abs(out.sum() - 1.0) > 0.05:
            raise RuntimeError(f"[{model_key}] output sum is {out.sum():.4f} — load failed")

        _LOADED_MODELS[model_key] = model
        print(f"✅ [{model_key}] ready")
        return model


def _preprocess_image(image_data: bytes):
    """
    Decode and resize only — the model's internal Lambda layer handles
    model-specific preprocessing (mobilenet_v2/densenet/efficientnet_v2).
    Send raw 0-255 values.
    """
    if not REAL_MODELS_AVAILABLE:
        return None
    img = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)          # keep as 0-255
    return tf.expand_dims(img, 0)


def get_available_models():
    """Return info on all available AI models (used by app.py endpoint)."""
    return {
        'mobilenetv2': {'name': 'MobileNetV2', 'description': 'Fast and lightweight.', 'accuracy': '89.24%'},
        'densenet201': {'name': 'DenseNet201', 'description': 'Deep and accurate.', 'accuracy': '91.60%'},
        'efficientnetv2s': {'name': 'EfficientNetV2S', 'description': 'Best balance of speed and accuracy.', 'accuracy': '93.70%'},
    }


import math

def sanitize_float(val, default=0.0) -> float:
    """Ensure float is valid number (not NaN or Inf) for clean JSON serialization."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def predict(model_key: str, image_bytes: bytes) -> dict:
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    start_time = time.time()

    if REAL_MODELS_AVAILABLE:
        try:
            model       = _get_model(model_key)
            img_tensor  = _preprocess_image(image_bytes)
            raw         = model(img_tensor, training=False)
            predictions = raw.numpy()[0]

            best_idx     = int(np.argmax(predictions))
            confidence   = sanitize_float(predictions[best_idx], 0.0)
            disease_name = CLASS_LABELS[best_idx]
            top_preds    = sorted(
                [{'disease_name': CLASS_LABELS[i], 'confidence': round(sanitize_float(s, 0.0) * 100, 2)}
                 for i, s in enumerate(predictions)],
                key=lambda x: x['confidence'], reverse=True
            )
        except Exception as e:
            print(f"Error running real model: {e}")
            disease_name, confidence, top_preds = "Error analyzing image", 0.0, []
    else:
        time.sleep(0.2)
        disease_name = "Bacterial Leaf Blight"
        confidence   = 0.85
        top_preds    = [
            {'disease_name': 'Bacterial Leaf Blight', 'confidence': 85.00},
            {'disease_name': 'Healthy Rice Leaf',     'confidence': 10.00},
            {'disease_name': 'Sheath Blight',         'confidence': 3.00},
            {'disease_name': 'Leaf Blast',            'confidence': 2.00},
        ]

    disease_id_map = {
        'Bacterial Leaf Blight': 1,
        'Leaf Blast':            2,
        'Sheath Blight':         4,
        'Healthy Rice Leaf':     100,
    }
    return {
        'disease_name':      disease_name,
        'disease_id':        disease_id_map.get(disease_name, 0),
        'confidence_score':  round(confidence * 100, 2),
        'inference_time_ms': int((time.time() - start_time) * 1000),
        'model_version':     f"{model_key}-v1.0",
        'real_model_loaded': REAL_MODELS_AVAILABLE,
        'top_predictions':   top_preds,
    }