"""Extract hard negatives from a folder using the TFLite model and copy them into hard_negatives/no/

Usage:
  python extract_hard_negatives.py --model models/model.tflite --src dataset_images/no --out hard_negatives/no --threshold 0.8
"""
import argparse
import os
import numpy as np
from PIL import Image
import shutil
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


def extract(model_path, src_dir, out_dir, threshold=0.8):
    os.makedirs(out_dir, exist_ok=True)
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    count = 0
    for fname in sorted(os.listdir(src_dir)):
        fpath = os.path.join(src_dir, fname)
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
        if prob > threshold:
            dst = os.path.join(out_dir, fname)
            shutil.copy2(fpath, dst)
            count += 1
    print(f'Copied {count} files with prob > {threshold} to {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/model.tflite')
    parser.add_argument('--src', default='dataset_images/no')
    parser.add_argument('--out', default='hard_negatives/no')
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()
    extract(args.model, args.src, args.out, args.threshold)

if __name__ == '__main__':
    main()
