import numpy as np
import cv2

_TF_AVAILABLE = None

def _ensure_tf():
    """Lazily import TensorFlow and cache availability."""
    global _TF_AVAILABLE
    if _TF_AVAILABLE is None:
        try:
            import tensorflow as tf
            _TF_AVAILABLE = tf
        except Exception:
            _TF_AVAILABLE = False
    return _TF_AVAILABLE

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Create a Grad-CAM heatmap for model's decisions.
    Args:
        img_array: Preprocessed image array (1, height, width, channels)
        model: Trained model
        last_conv_layer_name: Name of the last convolutional layer (auto-detected if None)
    Returns:
        Normalized heatmap array (height, width)
    """
    # Ensure TensorFlow is available at call time
    tf = _ensure_tf()
    if not tf:
        raise RuntimeError('TensorFlow is required for Grad-CAM but is not installed in this environment.')

    # Auto-detect last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("Could not find a convolutional layer in the model")

    grad_model = tf.keras.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        output = predictions[:, class_idx]

    grads = tape.gradient(output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

def create_gradcam_overlay(image, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.
    Args:
        image: Original image array (height, width, channels)
        heatmap: Grad-CAM heatmap array (height, width)
        alpha: Opacity of heatmap overlay
    Returns:
        Superimposed visualization image
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (width, height))
    
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Ensure image and jet colormap have same dimensions
    jet = cv2.resize(jet, (width, height))
    
    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(image, 1-alpha, jet, alpha, 0)
    
    return superimposed_img