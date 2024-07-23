import tensorflow as tf

# Load the HDF5 model
model = tf.keras.models.load_model('trained_model.h5')

# Recompile the model to ensure all attributes are set
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite model saved as model.tflite.")
