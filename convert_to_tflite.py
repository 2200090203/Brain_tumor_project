import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('models/final_model.h5')

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)