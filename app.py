from flask import Flask, request, render_template, jsonify, send_file, url_for, flash
import numpy as np
from PIL import Image
import cv2
import io
import os
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
import json

# Grad-CAM helper functions (project file)
from gradcam import make_gradcam_heatmap, create_gradcam_overlay
import logging

# Prefer the lightweight tflite_runtime if available, otherwise fall back to TensorFlow's lite
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    try:
        from tensorflow import lite as tflite
    except Exception:
        tflite = None


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configure basic logging
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger('mri_app')

# Configure for production
if os.environ.get('FLASK_ENV') == 'production':
    try:
        from flask_talisman import Talisman
        Talisman(app, force_https=True)
    except Exception:
        pass

# Create results/uploads directories if they don't exist
results_dir = os.path.join(os.path.dirname(__file__), 'static', 'results')
os.makedirs(results_dir, exist_ok=True)
uploads_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

# Globals for models
interpreter = None
interpreter_mri_filter = None
keras_model = None


def load_models(tflite_path='models/model.tflite', keras_path='models/final_model.h5', filter_path='models/mri_filter.tflite'):
    """Load main TFLite interpreter, optional MRI filter interpreter, and Keras model (for Grad-CAM).

    filter_path: optional TFLite model that scores how likely an image is an MRI. If present,
    it's used as a fast pre-check before running the main classifier.
    """
    global interpreter, interpreter_mri_filter, keras_model

    # Load TFLite interpreter(s)
    if tflite is None:
        raise RuntimeError('No TFLite interpreter available. Install tflite-runtime or TensorFlow.')

    # Main interpreter
    if os.path.exists(tflite_path):
        try:
            interpreter = tflite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
        except Exception as e:
            print('Warning: failed to load main TFLite model:', e)
            interpreter = None
    else:
        interpreter = None

    # MRI filter interpreter (optional)
    if filter_path and os.path.exists(filter_path):
        try:
            interpreter_mri_filter = tflite.Interpreter(model_path=filter_path)
            interpreter_mri_filter.allocate_tensors()
        except Exception as e:
            print('Warning: failed to load MRI filter TFLite model:', e)
            interpreter_mri_filter = None
    else:
        interpreter_mri_filter = None

    # Load Keras model for Grad-CAM (if available). Import tensorflow lazily to avoid hard dependency.
    if os.path.exists(keras_path):
        try:
            import tensorflow as tf
            keras_model = tf.keras.models.load_model(keras_path)
        except Exception as e:
            # If full TensorFlow isn't available, warn and continue without Grad-CAM support
            print('Warning: failed to import TensorFlow or load Keras model:', e)
            keras_model = None
    else:
        # If we don't have a full Keras model, some Grad-CAM functionality won't work
        keras_model = None

    print('Loaded models: main tflite=%s, filter_tflite=%s, keras=%s' % (
        tflite_path if interpreter is not None else 'None',
        filter_path if interpreter_mri_filter is not None else 'None',
        keras_path if keras_model is not None else 'None'))
    logger.info('Loaded models', extra={'main_tflite': tflite_path if interpreter is not None else None,
                                        'filter_tflite': filter_path if interpreter_mri_filter is not None else None,
                                        'keras': keras_path if keras_model is not None else None})


def load_model():
    """Compatibility wrapper for older entrypoints that expect `load_model()`.
    Calls `load_models()` under the hood.
    """
    return load_models()


def preprocess_image(image, target_size=(224, 224)):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize to match training size
    image = image.resize(target_size)
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, 0)
    return img_array


def run_tflite_on_pil_image(pil_image, interpreter_obj):
    """Run a TFLite interpreter on a PIL image and return a float score (0..1) or None on error."""
    try:
        input_details = interpreter_obj.get_input_details()
        output_details = interpreter_obj.get_output_details()

        # Infer target size from input_details
        in_shape = input_details[0]['shape']
        if len(in_shape) >= 3:
            target_h = int(in_shape[1])
            target_w = int(in_shape[2])
        else:
            target_h, target_w = 224, 224

        img = pil_image.convert('RGB').resize((target_w, target_h))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)

        # Handle quantized uint8 models
        if input_details[0]['dtype'] == np.uint8:
            inp = (arr * 255).astype(np.uint8)
        else:
            inp = arr.astype(input_details[0]['dtype'])

        interpreter_obj.set_tensor(input_details[0]['index'], inp)
        interpreter_obj.invoke()
        pred = interpreter_obj.get_tensor(output_details[0]['index'])
        # Interpret output: common shapes are (1,1) or (1,2)
        if pred.ndim == 2 and pred.shape[1] >= 2:
            score = float(pred[0][1])
        else:
            score = float(pred[0][0])
        return max(0.0, min(1.0, score))
    except Exception as e:
        print('TFLite filter inference error:', e)
        return None


def is_probable_mri(pil_image, target_size=(224, 224)):
    """Simple heuristic detector that estimates whether an input image looks like a brain MRI.

    This is intentionally lightweight (no extra ML model). It combines three signals:
    - low color saturation (most MRIs are effectively grayscale)
    - dark corners (MRI slices often have black background around the circular scan)
    - mid-range center brightness (brain tissue is not pure white nor pure black)

    Returns a float score in [0,1] (higher means more likely MRI).
    """
    try:
        img = pil_image.convert('RGB').resize(target_size)
        arr = np.array(img)
        # HSV saturation: low saturation suggests grayscale/medical image
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        mean_sat = float(np.mean(hsv[..., 1])) / 255.0

        # Grayscale center brightness
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        # corner darkness ratio (fraction of dark pixels in the corners)
        pad = max(8, min(h, w) // 16)
        corners = [gray[0:pad, 0:pad], gray[0:pad, w - pad:w], gray[h - pad:h, 0:pad], gray[h - pad:h, w - pad:w]]
        corner_darkness = np.mean([np.mean(c < 30) for c in corners])

        # center mean intensity normalized
        cx0, cx1 = h // 4, 3 * h // 4
        cy0, cy1 = w // 4, 3 * w // 4
        center = gray[cx0:cx1, cy0:cy1]
        center_mean = float(np.mean(center)) / 255.0

        # Compose a score (weights chosen conservatively from experiments on a few samples)
        score = (1.0 - mean_sat) * 0.55 + corner_darkness * 0.30 + (1.0 - abs(center_mean - 0.5) * 2) * 0.15
        # Clamp to [0,1]
        score = max(0.0, min(1.0, score))
        return score
    except Exception:
        # If anything goes wrong, return a low score to avoid false positives
        return 0.0


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded', 'success': False}), 400

        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))

        # First try the small MRI TFLite filter (fast). If it's not available or fails, fall back to heuristic.
        MRI_THRESHOLD = float(os.environ.get('MRI_SCORE_THRESHOLD', 0.45))
        filter_score = None
        if interpreter_mri_filter is not None:
            try:
                filter_score = run_tflite_on_pil_image(image, interpreter_mri_filter)
            except Exception as e:
                print('MRI filter run failed, falling back to heuristic:', e)

        if filter_score is None:
            # Fallback to heuristic
            filter_score = is_probable_mri(image)

        if filter_score < MRI_THRESHOLD:
            # Save the uploaded image for inspection, but don't run inference
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename or f'image_{timestamp}.png')
            filename_base, ext = os.path.splitext(original_filename)
            if ext == '':
                ext = '.png'
            saved_filename = f"{filename_base}_{timestamp}{ext}"
            image_path = os.path.join(uploads_dir, saved_filename)
            try:
                image.save(image_path)
            except Exception:
                pass
            return jsonify({
                'success': False,
                'error': 'Uploaded image does not appear to be a brain MRI',
                'mri_filter_score': float(filter_score),
                'threshold': MRI_THRESHOLD
            }), 400

        processed_image = preprocess_image(image)

        # Convert to OpenCV format for visualization
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(cv_image, (224, 224))

        # Ensure interpreter is loaded
        if interpreter is None:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare input according to expected dtype
        input_dtype = input_details[0]['dtype']
        input_data = processed_image
        if input_dtype == np.uint8:
            # If model expects uint8, scale to [0,255]
            input_data = (processed_image * 255).astype(np.uint8)
        else:
            input_data = processed_image.astype(input_dtype)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get prediction results
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Handle common output shapes (assume binary classification with two logits/probs)
        if prediction.ndim == 2 and prediction.shape[1] >= 2:
            probability = float(prediction[0][1])
        else:
            # Fallback: if single-output probability
            probability = float(prediction[0][0])

        # Generate Grad-CAM heatmap if Keras model is available
        heatmap = None
        if keras_model is not None:
            heatmap = make_gradcam_heatmap(processed_image, keras_model)
            gradcam_image = create_gradcam_overlay(cv_image, heatmap)
        else:
            # If we don't have a Keras model, just return the original resized image as overlay
            gradcam_image = cv_image

        # Convert visualization to base64 for sending to frontend
        _, buffer = cv2.imencode('.png', gradcam_image)
        gradcam_b64 = base64.b64encode(buffer).decode('utf-8')

        # Save the original image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename or f'image_{timestamp}.png')
        filename_base, ext = os.path.splitext(original_filename)
        if ext == '':
            ext = '.png'
        saved_filename = f"{filename_base}_{timestamp}{ext}"
        image_path = os.path.join(uploads_dir, saved_filename)
        image.save(image_path)

        # Save the heatmap visualization
        heatmap_filename = f"{filename_base}_{timestamp}_heatmap.png"
        heatmap_path = os.path.join(uploads_dir, heatmap_filename)
        cv2.imwrite(heatmap_path, gradcam_image)

        prediction_text = 'Tumor Detected' if probability > 0.5 else 'No Tumor Detected'
        confidence_level = 'High' if abs(probability - 0.5) > 0.35 else 'Moderate' if abs(probability - 0.5) > 0.2 else 'Low'

        # Save result to JSON file
        result = {
            'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'filename': saved_filename,
            'prediction': prediction_text,
            'probability': probability,
            'probability_percent': f"{probability * 100:.2f}%",
            'success': True,
            'confidence_level': confidence_level,
            'gradcam_image': f'data:image/png;base64,{gradcam_b64}',
            'analysis_details': {
                    'model_architecture': keras_model.name if keras_model is not None else 'tflite-only',
                    'input_shape': ( [int(x) for x in keras_model.input_shape[1:]] if keras_model is not None else [int(x) for x in input_details[0]['shape']] ),
                    'heatmap_intensity': float(np.mean(heatmap)) if heatmap is not None else None,
                    'region_of_interest': 'High attention regions shown in red' if heatmap is not None else None
                },
            'timestamp': datetime.now().isoformat()
        }

        result_file = os.path.join(results_dir, f"{result['id']}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


def _load_models_or_warn():
    """Helper to load models and print a warning on failure."""
    global interpreter, keras_model
    if interpreter is None:
        try:
            load_models()
        except Exception as e:
            # Log and continue; requests will see 'Model not loaded' if loading failed
            print('Warning: failed to load models on startup:', e)


# Register model loading on a Flask lifecycle event if available, otherwise load immediately.
if hasattr(app, 'before_first_request'):
    @app.before_first_request
    def load_models_before_first_request():
        _load_models_or_warn()
elif hasattr(app, 'before_serving'):
    # Flask >=2 provides before_serving; use async signature if present
    try:
        @app.before_serving
        async def load_models_before_serving():
            _load_models_or_warn()
    except Exception:
        # Fallback to synchronous registration
        @app.before_serving
        def load_models_before_serving():
            _load_models_or_warn()
else:
    # Last resort: load at import time (best-effort)
    try:
        _load_models_or_warn()
    except Exception:
        pass


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    # Load both models (TFLite for inference, Keras optional for Grad-CAM)
    try:
        load_models()
    except Exception as e:
        print('Warning: failed to load models at startup:', e)
    # Start Flask app on 0.0.0.0:8080 for container friendliness
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))