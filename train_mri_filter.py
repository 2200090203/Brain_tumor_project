import argparse
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf


def build_model(input_shape=(128, 128, 3)):
    base = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet', pooling='avg')
    x = base.output
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model


def main(args):
    batch_size = args.batch_size
    img_size = (args.img_size, args.img_size)
    data_dir = args.data_dir

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                       rotation_range=10, width_shift_range=0.05,
                                       height_shift_range=0.05, zoom_range=0.05,
                                       horizontal_flip=True)

    train_gen = train_datagen.flow_from_directory(data_dir, target_size=img_size,
                                                  batch_size=batch_size, class_mode='binary', subset='training')
    val_gen = train_datagen.flow_from_directory(data_dir, target_size=img_size,
                                                batch_size=batch_size, class_mode='binary', subset='validation', shuffle=False)

    model = build_model(input_shape=(args.img_size, args.img_size, 3))
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    os.makedirs('models', exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint('models/mri_filter.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    hist = model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=[ckpt])

    # Save SavedModel for TFLite conversion
    saved_dir = 'models/mri_filter_saved'
    model = tf.keras.models.load_model('models/mri_filter.h5')
    model.save(saved_dir, include_optimizer=False)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    try:
        tflite_model = converter.convert()
        with open('models/mri_filter.tflite', 'wb') as f:
            f.write(tflite_model)
        print('Saved tflite to models/mri_filter.tflite')
    except Exception as e:
        print('TFLite conversion failed:', e)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='dataset_images', help='root with yes/ and no/ subfolders')
    p.add_argument('--img_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=6)
    args = p.parse_args()
    main(args)
