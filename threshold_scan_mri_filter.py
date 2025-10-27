import os
import numpy as np
from PIL import Image
import argparse
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    from tensorflow import lite as tflite


def score_folder(model_path, folder):
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
            img = Image.open(fpath).convert('RGB')
        except Exception:
            continue
        # infer size
        in_shape = input_details[0]['shape']
        if len(in_shape) >= 3:
            h, w = int(in_shape[1]), int(in_shape[2])
        else:
            h, w = 224, 224
        img = img.resize((w, h))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        if input_details[0]['dtype'] == np.uint8:
            inp = (arr * 255).astype(np.uint8)
        else:
            inp = arr.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        if pred.ndim == 2 and pred.shape[1] >= 2:
            prob = float(pred[0][1])
        else:
            prob = float(pred[0][0])
        results.append((fname, prob))
    return results


def scan_thresholds(model_path, folder, thresholds):
    results = score_folder(model_path, folder)
    probs = np.array([p for (_, p) in results])
    n = len(probs)
    out = {'count': n, 'mean': float(np.mean(probs)) if n>0 else None}
    for t in thresholds:
        pass_count = int(np.sum(probs >= t))
        out[f'pass_ge_{t}'] = pass_count
        out[f'pass_rate_ge_{t}'] = pass_count / n if n>0 else None
    # Also return top 10
    idx = np.argsort(probs)[::-1][:10]
    topk = [(results[i][0], float(results[i][1])) for i in idx]
    out['top10'] = topk
    return out, results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/mri_filter.tflite')
    p.add_argument('--dir', required=True)
    p.add_argument('--thresholds', nargs='+', type=float, default=[0.3, 0.45, 0.5, 0.7])
    args = p.parse_args()
    out, results = scan_thresholds(args.model, args.dir, args.thresholds)
    print('Folder:', args.dir)
    print('Count:', out['count'], 'Mean:', out['mean'])
    for t in args.thresholds:
        print(f'Pass >= {t}: {out[f"pass_ge_{t}"]} ({out[f"pass_rate_ge_{t}"]:.2%})')
    print('\nTop 10:')
    for f,p in out['top10']:
        print(f, '->', f'{p:.6f}')
