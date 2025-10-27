import tensorflow as tf
import os
from train_mri_filter import build_model


def main():
    h5 = 'models/mri_filter.h5'
    saved_dir = 'models/mri_filter_saved'
    tflite_path = 'models/mri_filter.tflite'

    if not os.path.exists(h5):
        print('HDF5 model not found:', h5)
        return 1

    print('Loading', h5)
    loaded = tf.keras.models.load_model(h5)

    # Rebuild architecture and copy weights to a clean model to avoid save/export issues
    print('Rebuilding architecture and copying weights into a fresh model')
    fresh = build_model(input_shape=(128, 128, 3))
    try:
        fresh.set_weights(loaded.get_weights())
    except Exception as e:
        print('Failed to set weights on fresh model:', e)
        # fallback: use loaded model directly
        fresh = loaded

    # Export SavedModel
    if os.path.exists(saved_dir):
        print('SavedModel dir exists, removing and re-writing')
        import shutil
        shutil.rmtree(saved_dir)

    print('Exporting SavedModel to', saved_dir)
    try:
        # Keras 3 supports model.export for SavedModel export
        if hasattr(fresh, 'export'):
            fresh.export(saved_dir)
        else:
            # Older Keras: try saving as .keras then convert
            tmp_path = 'models/mri_filter.keras'
            fresh.save(tmp_path)
            # load and re-save with tf.saved_model if needed
            loaded_tmp = tf.keras.models.load_model(tmp_path)
            tf.saved_model.save(loaded_tmp, saved_dir)
    except Exception as e:
        print('Failed to export SavedModel:', e)
        return 2

    # Convert to TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print('Wrote tflite to', tflite_path)
    except Exception as e:
        print('TFLite conversion failed:', e)
        return 3

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
