"""Evaluate a TFLite model on a folder and report false-positive counts and top-k

Usage:
  python evaluate_tflite_counts.py --model models/model.tflite --dir dataset_images/no --topk 10

Prints counts for thresholds 0.5, 0.8, 0.9 and the top-k highest probabilities (filename, prob).
"""
import argparse
import os
import numpy as np
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    from tensorflow import lite as tflite


def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr


def evaluate(model_path, folder, topk=10):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    results = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath)
        except Exception:
            continue
        inp = preprocess_image(img)
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
        results.append((fname, prob))

    if not results:
        print('No images found or failed to evaluate.')
        return

    probs = np.array([p for (_, p) in results])
    n = len(probs)
    thresholds = [0.5, 0.8, 0.9]
    print('\nSummary:')
    print('count:', n)
    print('mean prob:', float(np.mean(probs)))
    print('median prob:', float(np.median(probs)))
    print('min prob:', float(np.min(probs)), 'max prob:', float(np.max(probs)))
    for t in thresholds:
        c = int(np.sum(probs > t))
        print(f'count > {t}: {c} ({c/n:.4%})')

    # Top-k
    topk = min(topk, n)
    idx = np.argsort(probs)[::-1][:topk]
    print(f'\nTop {topk} highest probabilities:')
    for i in idx:
        print(results[i][0], '->', f'{results[i][1]:.6f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/model.tflite')
    parser.add_argument('--dir', default='dataset_images/no')
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()
    evaluate(args.model, args.dir, args.topk)


if __name__ == '__main__':
    main()
