import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

def load_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image."""
    img = keras_image.load_img(image_path, target_size=target_size)
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize to [0,1] range
    return img

def load_class_labels(filename):
    """Load class labels from a file."""
    with open(filename, 'r') as f:
        class_labels = {int(line.split(': ')[0]): line.split(': ')[1].strip() for line in f}
    return class_labels

def main():
    # Path to the model, labels, and image
    model_path = 'trained_model.h5'
    class_labels_path = 'classes.txt'
    image_path = 'image5.jpg'  # Specify the path to your image here
    
    # Load the class labels
    class_labels = load_class_labels(class_labels_path)
    print("Class labels loaded:", class_labels)
    
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")

    # Load and preprocess the image
    img = load_image(image_path)
    
    # Run inference
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_class_index]
    
    print(f"Image: {image_path}, Predicted label: {predicted_label}")

if __name__ == '__main__':
    main()
