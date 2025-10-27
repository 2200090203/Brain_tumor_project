# dataset.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(dataset_dir, img_size=(224,224), batch_size=32, val_split=0.2):
    """
    Expects dataset_dir structured like:
    dataset_dir/
      class_1/
      class_2/
      ...
    Returns: train_generator, val_generator
    
    Note: Images are loaded as grayscale (1 channel) and converted to RGB in the model
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        brightness_range=[0.7, 1.3],
        validation_split=val_split
    )
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)
    
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        color_mode='rgb'  # Convert to RGB
    )
    val_generator = test_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='rgb'  # Convert to RGB
    )
    return train_generator, val_generator