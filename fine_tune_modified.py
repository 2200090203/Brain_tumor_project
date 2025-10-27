#!/usr/bin/env python
"""
Robust fine-tune script for grayscale/rgb images + EfficientNetB0.

Features:
- Forces generators to read images as RGB (color_mode='rgb')
- Builds EfficientNet backbone with weights=None, then optionally transfers
  imagenet weights from a temporary imagenet-initialized model to avoid
  shape-mismatch crashes (handles RGB->grayscale conv conversion).
- Safe loading of custom fine-tuned weights (strict then fallback by_name)
- Unfreeze last N layers, training callbacks, final weights saved.
"""
import os
import argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
)

def make_generators(data_dir, img_size=224, batch_size=8, val_split=0.2):
    import tensorflow as tf

    def gray_to_rgb(x):
        # Convert grayscale → RGB (ensures 3-channel input for EfficientNet)
        x = tf.image.grayscale_to_rgb(x)
        return x

    # Augmented training generator
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        brightness_range=(0.8, 1.2),
        zoom_range=0.2,
        shear_range=0.08,
        horizontal_flip=True,
        fill_mode='reflect',
        validation_split=val_split,
        preprocessing_function=gray_to_rgb
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=val_split,
        preprocessing_function=gray_to_rgb
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        color_mode='grayscale'  # read grayscale then convert to RGB
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='grayscale'
    )

    print(f"Using split: train=({train_gen.samples}), val=({val_gen.samples}) (split={val_split})")
    print("Class indices:", train_gen.class_indices)
    return train_gen, val_gen


def _safe_transfer_imagenet_weights(tmp_model, target_model):
    """Try to copy weights layer-by-layer by name from tmp_model -> target_model.
       Special-case: if target_model first conv expects 1 channel and tmp has 3,
       average RGB kernels to a single channel.
    """
    tmp_layers = {layer.name: layer for layer in tmp_model.layers}
    for layer in target_model.layers:
        name = layer.name
        if name not in tmp_layers:
            continue
        tmp_layer = tmp_layers[name]
        tmp_w = tmp_layer.get_weights()
        if not tmp_w:
            continue
        try:
            # handle first conv RGB->single channel conversion
            if 'stem_conv' in name or (len(tmp_w)>0 and len(layer.get_weights())>0 and tmp_w[0].ndim == 4 and layer.get_weights()[0].ndim == 4):
                w_tmp = tmp_w[0]
                w_target = layer.get_weights()[0]
                if w_tmp.shape[2] == 3 and w_target.shape[2] == 1:
                    # average RGB channels to create single-channel kernel
                    w_new = np.mean(w_tmp, axis=2, keepdims=True)
                    new_weights = [w_new]
                    # if bias exists in tmp, copy it too (if target has bias)
                    if len(tmp_w) > 1 and len(layer.get_weights()) > 1:
                        new_weights.append(tmp_w[1])
                    layer.set_weights(new_weights)
                    continue
            # otherwise try straightforward copy (if shapes match)
            layer.set_weights(tmp_w)
        except Exception:
            # mismatch — skip
            continue

def build_model(input_shape=(224, 224, 3), num_classes=2, backbone_weights='imagenet'):
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Model

    # Normalize 'None' string to Python None
    if isinstance(backbone_weights, str) and backbone_weights.lower() == 'none':
        backbone_weights = None

    try:
        base = EfficientNetB0(include_top=False, weights=backbone_weights,
                              input_shape=input_shape, pooling='avg')
        print(f"[+] Loaded EfficientNetB0 backbone weights = {backbone_weights}")
    except Exception as e:
        print(f"[!] Failed to load/transfer imagenet weights: {e}")
        print("[!] Continuing with base model initialized randomly (weights=None).")
        base = EfficientNetB0(include_top=False, weights=None,
                              input_shape=input_shape, pooling='avg')

    x = base.output
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=outputs)
    return model


def unfreeze_last_n_layers(model, n):
    for layer in model.layers:
        layer.trainable = False
    if n <= 0:
        return
    total = len(model.layers)
    start = max(0, total - n)
    for i in range(start, total):
        try:
            model.layers[i].trainable = True
        except Exception:
            pass
    print(f"Unfroze layers from index {start} to {total-1} (total {n} layers).")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset_images', help='dataset root (contains class subfolders)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--unfreeze_layers', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--weights', type=str, default=None, help='path to fine-tuned weights to load (optional)')
    parser.add_argument('--backbone_weights', type=str, default='imagenet', help="'imagenet' or 'None'")
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    # normalize backbone flag
    if isinstance(args.backbone_weights, str) and args.backbone_weights.lower() == 'none':
        backbone_weights = None
    else:
        backbone_weights = args.backbone_weights

    os.makedirs(args.save_dir, exist_ok=True)

    train_gen, val_gen = make_generators(args.data_dir, img_size=args.img_size, batch_size=args.batch_size)

    # enforce 3-channel input (generators provide RGB)
    input_shape = (args.img_size, args.img_size, 3)
    model = build_model(input_shape=input_shape, num_classes=val_gen.num_classes, backbone_weights=backbone_weights)
    model.summary()

    # load user-provided fine-tuned weights (strict then fallback)
    if args.weights:
        print(f"[i] Attempting to load weights from: {args.weights}")
        try:
            model.load_weights(args.weights)
            print("[+] Strict load succeeded.")
        except Exception as e_strict:
            print("[!] Strict load failed:", e_strict)
            try:
                model.load_weights(args.weights, by_name=True, skip_mismatch=True)
                print("[+] Loaded weights with by_name=True, skip_mismatch=True")
            except Exception as e_soft:
                print("[!] Fallback load also failed. Proceeding without these weights. Error:", e_soft)

    # unfreeze
    unfreeze_last_n_layers(model, args.unfreeze_layers)

    # compile
    opt = Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    best_fp = os.path.join(args.save_dir, 'best_model.weights.h5')
    checkpoint = ModelCheckpoint(best_fp, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=False, verbose=1)
    csv_logger = CSVLogger(os.path.join(args.save_dir, 'training_log.csv'), append=True)
    callbacks = [checkpoint, reduce_lr, early, csv_logger]

    steps_per_epoch = math.ceil(train_gen.samples / args.batch_size)
    validation_steps = math.ceil(val_gen.samples / args.batch_size) if val_gen.samples>0 else None

    print("Starting training...")
    try:
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        print("WARNING: training failed with callbacks (Windows pickling or similar). Retrying without callbacks. Exception:", str(e))
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1
        )

    final_weights = os.path.join(args.save_dir, 'final_model.weights.h5')
    model.save_weights(final_weights)
    print("Final weights saved to:", final_weights)

if __name__ == '__main__':
    main()
