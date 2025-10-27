# train.py — transfer-learning training script (categorical-safe)
import argparse, os
import numpy as np
from dataset import create_generators
from model import build_model
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def compute_class_weights(generator):
    labels = generator.classes
    classes = np.unique(labels)
    cw = class_weight.compute_class_weight('balanced', classes=classes, y=labels)
    return {int(i): float(w) for i, w in enumerate(cw)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='path to dataset folder')
    parser.add_argument('--img_size', type=int, default=224, help='image size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=15, help='epochs')
    parser.add_argument('--save_dir', type=str, default='models', help='where to save models and figs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    train_gen, val_gen = create_generators(args.data_dir, img_size=(args.img_size,args.img_size), batch_size=args.batch_size)
    num_classes = len(train_gen.class_indices)
    print("Class indices:", train_gen.class_indices, "num_classes:", num_classes)

    model = build_model(input_shape=(args.img_size, args.img_size, 3), num_classes=num_classes)

    # Use categorical loss when generator gives one-hot vectors (num_classes >= 2)
    if num_classes >= 2:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        loss = 'binary_crossentropy'
        metrics = ['accuracy']

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # compute class weights
    cw = compute_class_weights(train_gen)
    print("Class weights:", cw)

    checkpoint_path = os.path.join(args.save_dir, 'best_model.weights.h5')
    callbacks = [
        ModelCheckpoint(
            checkpoint_path, 
            monitor='val_accuracy', 
            save_best_only=True, 
            verbose=1,
            save_weights_only=True  # Save only weights for better compatibility
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # More aggressive reduction
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # More patience
            verbose=1,
            restore_best_weights=True
        )
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks, class_weight=cw)
    final_path = os.path.join(args.save_dir, 'final_model.h5')
    model.save(final_path)
    print("Saved final model to:", final_path)

if __name__ == '__main__':
    main()
