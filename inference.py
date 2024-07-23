import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

def load_labels(label_file):
    """Load class labels from a file."""
    label_to_index = {}
    with open(label_file, 'r') as f:
        for line in f:
            index, label = line.strip().split(': ')
            label_to_index[int(index)] = label  # Ensure index is integer
    return label_to_index

def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def load_model(model_path):
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)

def predict(model, image_array, index_to_label):
    """Make a prediction on the preprocessed image."""
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions, axis=-1)[0]
    predicted_label = index_to_label.get(predicted_index, "Unknown")
    return predicted_index, predicted_label

def main():
    # Paths
    model_path = "trained_model.keras"
    class_labels_path = "class_labels.txt"
    test_dir = "dataset/training"  # Update this with the path to your test images folder

    # Load labels
    label_to_index = load_labels(class_labels_path)
    print(f"Loaded labels: {label_to_index}")
    index_to_label = {index: label for index, label in label_to_index.items()}

    # Load the model
    model = load_model(model_path)

    # Initialize counts and results
    class_counts = {}
    results = []

    # Iterate over the images in the test folder
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                class_name = file.split('.')[0]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                
                if class_counts[class_name] >= 2:
                    continue

                image_path = os.path.join(root, file)
                try:
                    # Preprocess the image
                    image_array = preprocess_image(image_path)
                    # Make a prediction
                    predicted_index, predicted_label = predict(model, image_array, index_to_label)
                    true_label = class_name
                    results.append((file, true_label, predicted_label))
                    print(f"Image: {file}, True Label: {true_label}, Predicted Label: {predicted_label}")
                    class_counts[class_name] += 1
                except Exception as e:
                    print(f"Error processing image {file}: {e}")

    # Print a summary of the results
    print("\nSummary of predictions:")
    for file, true_label, predicted_label in results:
        print(f"Image: {file}, True Label: {true_label}, Predicted Label: {predicted_label}")

if __name__ == '__main__':
    main()
