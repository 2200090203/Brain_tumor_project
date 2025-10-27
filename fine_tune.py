# fine_tune.py
import os
import argparse
import traceback
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to base model weights (h5/keras)")
    p.add_argument("--data_dir", required=True, help="Root folder with either train/ & val/ subfolders OR class subfolders")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--unfreeze_layers", type=int, default=50, help="How many last layers to unfreeze")
    p.add_argument("--save_dir", default="models", help="Where to save fine-tuned weights/model")
    p.add_argument("--save_full_model", action="store_true", help="Also save full model in .keras format at the end")
    return p

def make_generators(data_dir, img_size, batch_size, val_split=0.2):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        # explicit train/val folders found
        train_gen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        ).flow_from_directory(
            train_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True
        )
        val_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
            val_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False
        )
        print(f"Using explicit folders: train=({train_gen.samples} imgs), val=({val_gen.samples} imgs)")
    else:
        # Fall back to single folder with class subfolders; use validation_split
        if not os.path.isdir(data_dir):
            raise ValueError(f"Provided data_dir does not exist: {data_dir}")

        print("No explicit train/val subfolders found — using `validation_split` on the provided data_dir.")
        base_gen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=val_split
        )

        train_gen = base_gen.flow_from_directory(
            data_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True
        )

        val_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
            data_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False
        )

        print(f"Using split: train=({train_gen.samples} imgs), val=({val_gen.samples} imgs) (split={val_split})")

    if len(train_gen.class_indices) == 0:
        raise ValueError("No classes found in data. Ensure `data_dir` contains class subfolders with images.")

    print("Loaded generators. Class indices:", train_gen.class_indices)
    return train_gen, val_gen

def load_model_from_file(weights_path, img_size, num_classes=2):
    # try to load a full saved model first
    try:
        model = keras.models.load_model(weights_path)
        print("Loaded full model from:", weights_path)
        return model
    except Exception as e:
        print(f"Could not load full model from {weights_path} (will try to load weights into a new model). Error: {e}")

    # fallback: build MobileNetV3Large-based classifier (matches layer names seen in your log)
    from tensorflow.keras.applications import MobileNetV3Large
    base = MobileNetV3Large(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    x = base.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=base.input, outputs=out)

    # try loading weights (may be just top weights or whole-model weights)
    try:
        model.load_weights(weights_path)
        print("Loaded weights into MobileNetV3Large-based model from:", weights_path)
    except Exception as e:
        print("Warning: could not load weights into fallback model. Proceeding with model (some layers may be randomly initialized).")
        print("Load error:", e)

    return model

def unfreeze_last_n_layers(model, n):
    # freeze everything first
    for layer in model.layers:
        layer.trainable = False

    # unfreeze last n layers that have weights
    cnt = 0
    for layer in reversed(model.layers):
        if len(layer.weights) == 0:
            continue
        layer.trainable = True
        cnt += 1
        if cnt >= n:
            break

    trainable_params = np.sum([np.prod(w.shape) for w in model.trainable_weights]) if model.trainable_weights else 0
    print(f"Unfroze last {cnt} layers. Trainable params count: {trainable_params}")
    return model

def main():
    args = build_parser().parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_gen, val_gen = make_generators(args.data_dir, args.img_size, args.batch_size)

    model = load_model_from_file(args.weights, args.img_size, num_classes=len(train_gen.class_indices))

    model = unfreeze_last_n_layers(model, args.unfreeze_layers)

    lr = 1e-5
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Model compiled for fine-tuning with lr=", lr)

    checkpoint_path = os.path.join(args.save_dir, "best_model.weights.h5")
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=6, verbose=1)

    callbacks = [checkpoint, reduce_lr, early_stop]

    steps_per_epoch = int(np.ceil(train_gen.samples / args.batch_size))
    validation_steps = int(np.ceil(val_gen.samples / args.batch_size))

    # safe fit wrapper (handles Windows pickling issues with callbacks)
    try:
        print("Starting training with callbacks.")
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
    except Exception as e:
        tb = traceback.format_exc()
        if "cannot pickle" in tb or "cannot pickle 'module' object" in tb or "TypeError: cannot pickle" in tb:
            print("⚠️ Pickle/deepcopy error detected with callbacks on this platform. Retrying without callbacks.")
            print("Full exception:\n", tb)
            # retry without callbacks
            history = model.fit(
                train_gen,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=None
            )
            # try to save weights after training
            try:
                model.save_weights(checkpoint_path)
                print("Saved weights after fallback training to:", checkpoint_path)
            except Exception as e2:
                print("Could not save weights after fallback training:", e2)
        else:
            print("Unexpected training error; re-raising.")
            raise

    # always save final weights
    final_weights_path = os.path.join(args.save_dir, "final_weights.h5")
    try:
        model.save_weights(final_weights_path)
        print("Saved final weights to:", final_weights_path)
    except Exception as e:
        print("Could not save final weights:", e)

    if args.save_full_model:
        keras_path = os.path.join(args.save_dir, "final_model.keras")
        try:
            model.save(keras_path)
            print("Saved full model to:", keras_path)
        except Exception as e:
            print("Failed saving full model to .keras; attempting HDF5 fallback.")
            try:
                fallback = os.path.join(args.save_dir, "final_model.h5")
                model.save(fallback)
                print("Saved full model (HDF5) to:", fallback)
            except Exception as e2:
                print("Failed to save full model in either format:", e2)

if __name__ == "__main__":
    main()
