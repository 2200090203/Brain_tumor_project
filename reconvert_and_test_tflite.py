"""Convert a saved full Keras model to TFLite, test predictions vs Keras on sample images, and run evaluator.

Usage:
  python reconvert_and_test_tflite.py --keras models/finetuned_full_model.h5 --out models/finetuned_fixed.tflite
"""
import argparse
import sys
import subprocess
from PIL import Image
import numpy as np
import tensorflow as tf


def convert(keras_path, out_path):
    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    tflite_model = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print('Saved', out_path)
    return model


def predict_keras(model, img_path):
    img = Image.open(img_path).convert('RGB').resize((224,224))
    arr = np.array(img, dtype=np.float32)/255.0
    arr = np.expand_dims(arr,0)
    return model.predict(arr)


def predict_tflite(tflite_path, img_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = Image.open(img_path).convert('RGB').resize((224,224))
    arr = np.array(img, dtype=np.float32)/255.0
    inp = arr.astype(input_details[0]['dtype'])
    if input_details[0]['dtype'] == np.uint8:
        inp = (arr*255).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(inp,0))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras', default='models/finetuned_full_model.h5')
    parser.add_argument('--out', default='models/finetuned_fixed.tflite')
    parser.add_argument('--pos_sample', default='dataset_images/yes/y0.jpg')
    parser.add_argument('--neg_sample', default='dataset_images/no/no520.jpg')
    parser.add_argument('--dir', default='dataset_images/no')
    args = parser.parse_args()

    model = convert(args.keras, args.out)
    kp_pos = predict_keras(model, args.pos_sample)
    kp_neg = predict_keras(model, args.neg_sample)
    print('Keras pos:', kp_pos, 'Keras neg:', kp_neg)

    tp_pos = predict_tflite(args.out, args.pos_sample)
    tp_neg = predict_tflite(args.out, args.neg_sample)
    print('TFLite pos:', tp_pos, 'TFLite neg:', tp_neg)

    # run auto-index evaluator
    cmd = [sys.executable, 'evaluate_tflite_autoidx.py', '--model', args.out, '--pos_dir', 'dataset_images/yes', '--dir', args.dir, '--topk', '10']
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
