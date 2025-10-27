# evaluate_model.py
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf


# try seaborn for nicer heatmap; fallback to matplotlib if not available
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

def build_model(input_shape=(224, 224, 3), num_classes=2, backbone_weights=None):
    """
    Build the same model architecture we used for training
    """
    inputs = layers.Input(shape=input_shape)
    
    # First conv block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Second conv block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Third conv block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_val_generator(data_dir, img_size=224, batch_size=8, validation_split=0.2):
    """
    Creates a validation generator that reads images and returns RGB tensors.
    If files are grayscale, Pillow may open them as L; using color_mode='rgb'
    asks Keras to return 3-channel images (R=G=B).
    """
    datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='rgb'   # <-- read as 3-channel RGB
    )
    return val_gen

def try_load_weights(model, weights_path):
    """
    Try to load weights robustly. If file contains a full model vs weights-only,
    attempt multiple load strategies.
    """
    if not weights_path or not os.path.exists(weights_path):
        print(f"[!] Weights path not provided or does not exist: {weights_path}")
        return False

    print(f"[i] Attempting to load weights from: {weights_path}")
    # First try weights-only load (typical if ModelCheckpoint(save_weights_only=True))
    try:
        model.load_weights(weights_path)
        print("[+] Loaded weights via model.load_weights(weights_path)")
        return True
    except Exception as e:
        print(f"[-] model.load_weights failed: {e}")

    # Try loading as a full model file (HDF5) then transfer weights by name (if possible)
    try:
        print("[i] Trying to load as full model file (tf.keras.models.load_model)...")
        loaded = tf.keras.models.load_model(weights_path)
        print("[+] Full model loaded successfully. Now transferring weights by name (skip_mismatch=True).")
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        return True
    except Exception as e:
        print(f"[-] load_model / by_name transfer failed: {e}")

    # As a last resort: try loading weights into the base by name skipping mismatches
    try:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("[+] Loaded weights with by_name=True, skip_mismatch=True")
        return True
    except Exception as e:
        print(f"[-] Final fallback load failed: {e}")

    return False

def plot_confusion_matrix(cm, class_names, title="Confusion matrix"):
    plt.figure(figsize=(6, 5))
    if _HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset_images', help='dataset directory (train/val or root with split)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weights', type=str, default='models/best_model.weights.h5', help='path to weights (weights-only or model file)')
    parser.add_argument('--backbone_weights', type=str, default='imagenet', help="weights for backbone: 'imagenet' or None")
    args = parser.parse_args()

    # ---------- normalize --backbone_weights so string "None" becomes Python None ----------
    if isinstance(args.backbone_weights, str) and args.backbone_weights.lower() == "none":
        args.backbone_weights = None

    # now create the validation generator (reads grayscale as RGB or uses color_mode='rgb')
    val_gen = get_val_generator(args.data_dir, img_size=args.img_size, batch_size=args.batch_size)

    # build model with 3-channel input (RGB)
    model = build_model(input_shape=(args.img_size, args.img_size, 3),
                        num_classes=val_gen.num_classes,
                        backbone_weights=args.backbone_weights)

    # Try loading custom fine-tuned weights (robustly)
    loaded = try_load_weights(model, args.weights)
    if not loaded:
        print("[!] Could not load fine-tuned weights; the model will use backbone initial weights (imagenet or random).")

    # Compile for evaluation (metrics)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Evaluate
    print("[i] Running evaluation on validation set...")
    steps = int(np.ceil(val_gen.samples / float(args.batch_size)))
    results = model.evaluate(val_gen, steps=steps, verbose=1)
    print(f"[i] Eval results (loss, accuracy): {results}")

    # Predict and produce report
    print("[i] Predicting on validation data...")
    val_gen.reset()
    preds = model.predict(val_gen, steps=steps, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes  # because shuffle=False

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    plot_confusion_matrix(cm, class_names=list(val_gen.class_indices.keys()))

if __name__ == '__main__':
    main()
