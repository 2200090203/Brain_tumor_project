"""Simple utility to run the TFLite model over a folder of images.

Usage:
  python evaluate_tflite_on_folder.py --model models/model.tflite --dir dataset_images/other_scans

This prints per-image probabilities and a small summary.
"""
import argparse
import os
import numpy as np
from PIL import Image
import json
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    from tensorflow import lite as tflite


def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # PIL expects (width, height)
    image = image.resize((target_size[0], target_size[1]))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr


def run_folder(model_path, folder):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    results = []
    # Determine required input size from the model if possible
    try:
        in_shape = input_details[0]['shape']
        # shape is (1, H, W, C) or (1, W, H, C) depending on model; assume (1,H,W,C)
        if len(in_shape) >= 3:
            target_size = (int(in_shape[1]), int(in_shape[2]))
        else:
            target_size = (224, 224)
    except Exception:
        target_size = (224, 224)

    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath)
        except Exception as e:
            print('Skipping', fname, '(', e, ')')
            continue
        inp = preprocess_image(img, target_size=target_size)
        # Handle uint8 input models
        if input_details[0]['dtype'] == np.uint8:
            inp = (inp * 255).astype(np.uint8)
        else:
            inp = inp.astype(input_details[0]['dtype'])

        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        if pred.ndim == 2 and pred.shape[1] >= 2:
            prob = float(pred[0][1])
        else:
            prob = float(pred[0][0])
        results.append({'file': fname, 'probability': prob})
        print(fname, '->', f'{prob:.4f}')

    # Summary
    probs = [r['probability'] for r in results]
    if probs:
        print('\nSummary:')
        print('count:', len(probs))
        print('mean prob:', np.mean(probs))
        print('median prob:', np.median(probs))
        print('min prob:', np.min(probs), 'max prob:', np.max(probs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/model.tflite')
    parser.add_argument('--dir', default='dataset_images')
    args = parser.parse_args()
    run_folder(args.model, args.dir)


if __name__ == '__main__':
    main()
