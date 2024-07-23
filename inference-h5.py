import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('model/model.h5')

# Print model summary
model.summary()

# Class mapping
class_mapping = {
    0: 'black_dress',
    1: 'black_pants',
    2: 'black_shirt',
    3: 'black_shoes',
    4: 'black_shorts',
    5: 'blue_dress',
    6: 'blue_pants',
    7: 'blue_shirts',
    8: 'blue_shorts',
    9: 'red_dress',
    10: 'red_pants',
    11: 'red_shoes',
    12: 'white_dress',
    13: 'white_pants',
    14: 'white_shorts'
}

# Function to preprocess the input image
def preprocess_input_image(image_path):
    # Load the image
    image = Image.open(image_path)
    # Resize the image to the expected input shape of the model
    image = image.resize((96, 96))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image
    image = image.astype('float32') / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Path to the input image
image_path = 'test/image3.jpg'

# Preprocess the input image
preprocessed_image = preprocess_input_image(image_path)

# Make prediction
prediction = model.predict(preprocessed_image)

# Get the predicted class index
predicted_class_index = np.argmax(prediction, axis=1)[0]

# Map the predicted class index to the class name
predicted_class_name = class_mapping[predicted_class_index]

# Print the prediction
print('Prediction:', predicted_class_name)
