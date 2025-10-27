"""Evaluate a Keras model (weights or full model) on a folder and report counts/top-k.

Usage:
  python evaluate_keras_on_folder.py --weights models/finetune_quick/best_model.weights.h5 --dir dataset_images/no
"""
import argparse
import os
import numpy as np
from PIL import Image
from fine_tune import load_model_from_file


def preprocess_image(image, target_size=(224,224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = np.array(image, dtype=np.float32) / 255.0
    return arr


def evaluate(weights_path, folder, img_size=224, topk=10):
    model = load_model_from_file(weights_path, img_size, num_classes=2)
    # model may expect batch dimension
    results = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath)
        except Exception:
            continue
        arr = preprocess_image(img, target_size=(img_size,img_size))
        inp = np.expand_dims(arr, 0)
        pred = model.predict(inp)
        if pred.ndim == 2 and pred.shape[1] >= 2:
            prob = float(pred[0][1])
        else:
            prob = float(pred[0][0])
        results.append((fname, prob))
        print(fname, '->', f'{prob:.4f}')

    if not results:
        print('No results')
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

    topk = min(topk, n)
    idx = np.argsort(probs)[::-1][:topk]
    print(f'\nTop {topk} highest probabilities:')
    for i in idx:
        print(results[i][0], '->', f'{results[i][1]:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='models/finetune_quick/best_model.weights.h5')
    parser.add_argument('--dir', default='dataset_images/no')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()
    evaluate(args.weights, args.dir, args.img_size, args.topk)
