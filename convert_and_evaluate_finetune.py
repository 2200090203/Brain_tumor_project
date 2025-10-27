"""Load fine-tuned weights, export a full Keras model, convert to TFLite, and evaluate on a folder.

Usage:
  python convert_and_evaluate_finetune.py --weights models/finetune_quick/best_model.weights.h5 --out_tflite models/finetuned.tflite --dir dataset_images/no
"""
import argparse
import os
import subprocess
import sys

import tensorflow as tf

# import helper from fine_tune.py
from fine_tune import load_model_from_file


def convert_to_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Use default optimizations (can be changed later)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print('TFLite conversion failed with Optimize.DEFAULT, retrying without optimizations. Error:', e)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print('Saved TFLite model to', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='models/finetune_quick/best_model.weights.h5')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--out_keras', default='models/finetuned_full_model.h5')
    parser.add_argument('--out_tflite', default='models/finetuned.tflite')
    parser.add_argument('--dir', default='dataset_images/no')
    args = parser.parse_args()

    weights = args.weights
    img_size = args.img_size

    print('Loading model architecture and applying weights from', weights)
    model = load_model_from_file(weights, img_size, num_classes=2)

    # Save full model
    print('Saving full Keras model to', args.out_keras)
    try:
        model.save(args.out_keras)
        print('Saved full Keras model.')
    except Exception as e:
        print('Could not save full model to', args.out_keras, 'Error:', e)

    # Convert to TFLite
    print('Converting to TFLite ->', args.out_tflite)
    convert_to_tflite(model, args.out_tflite)

    # Run evaluator
    cmd = [sys.executable, 'evaluate_tflite_on_folder.py', '--model', args.out_tflite, '--dir', args.dir]
    print('Running evaluator:', ' '.join(cmd))
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
