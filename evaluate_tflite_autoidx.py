"""Evaluate a TFLite model on a folder but auto-detect which output index corresponds to the 'yes' (tumor) class.

Strategy:
- Load the TFLite model and run a single known-positive MRI image from --pos_sample (defaults to first image in dataset_images/yes).
- If that sample scores higher on index 0 than index 1, we assume index 0==tumor_index; otherwise index 1.
- Then evaluate the folder using the detected tumor_index and report counts and top-k.

Usage:
  python evaluate_tflite_autoidx.py --model models/finetuned.tflite --pos_dir dataset_images/yes --dir dataset_images/no --topk 10
"""
import argparse
import os
import numpy as np
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    from tensorflow import lite as tflite


def preprocess_image(image, target_size=(224,224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr


def detect_tumor_index(interpreter, sample_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = Image.open(sample_path)
    inp = preprocess_image(img)
    if input_details[0]['dtype'] == np.uint8:
        inp = (inp * 255).astype(np.uint8)
    else:
        inp = inp.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    pred = pred.flatten()
    # choose argmax as tumor index
    tumor_idx = int(np.argmax(pred))
    print(f'Detected tumor index={tumor_idx} from sample {os.path.basename(sample_path)} -> pred={pred}')
    return tumor_idx


def evaluate(model_path, folder, pos_dir=None, topk=10):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # detect sample
    if pos_dir:
        # pick first file
        samples = [f for f in sorted(os.listdir(pos_dir)) if os.path.isfile(os.path.join(pos_dir, f))]
        if samples:
            sample = os.path.join(pos_dir, samples[0])
            tumor_idx = detect_tumor_index(interpreter, sample)
        else:
            tumor_idx = 1
    else:
        tumor_idx = 1

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
        pred = pred.flatten()
        prob = float(pred[tumor_idx])
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
    parser.add_argument('--model', default='models/finetuned.tflite')
    parser.add_argument('--pos_dir', default='dataset_images/yes')
    parser.add_argument('--dir', default='dataset_images/no')
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()
    evaluate(args.model, args.dir, args.pos_dir, args.topk)
