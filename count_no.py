"""Count high-confidence positives in dataset_images/no using the TFLite model
"""
import os
import numpy as np
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    from tensorflow import lite as tflite

MODEL_PATH = 'models/model.tflite'
FOLDER = 'dataset_images/no'


def preprocess_image(image, target_size=(224,224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr


def main():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    probs = []
    files = []
    for fname in sorted(os.listdir(FOLDER)):
        fpath = os.path.join(FOLDER, fname)
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
        probs.append(prob)
        files.append(fname)

    pairs = list(zip(files, probs))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

    total = len(probs)
    print('\nSummary:')
    print('total files:', total)
    for thr in (0.5, 0.8, 0.9):
        cnt = sum(1 for p in probs if p > thr)
        print(f'> {thr}: {cnt}')
    if probs:
        print('mean:', np.mean(probs))
        print('median:', np.median(probs))
        print('min:', np.min(probs), 'max:', np.max(probs))

    print('\nTop 10 highest probabilities:')
    for f, p in pairs_sorted[:10]:
        print(f, '->', f'{p:.4f}')


if __name__ == '__main__':
    main()
