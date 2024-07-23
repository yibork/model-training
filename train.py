import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def extract_label_from_filename(filename):
    """Extract class label from the filename."""
    label = filename.split('.')[0]
    return label

def load_data(data_dir, max_images_per_class=20):
    """Load images and labels from a directory, up to a maximum number of images per class."""
    images = []
    labels = []
    class_count = defaultdict(int)
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                label = extract_label_from_filename(file)
                if class_count[label] < max_images_per_class:
                    image_path = os.path.join(root, file)
                    img = keras_image.load_img(image_path, target_size=(224, 224))
                    img = keras_image.img_to_array(img)
                    images.append(img)
                    labels.append(label)
                    class_count[label] += 1
                if len(class_count) >= 14 and all(count >= max_images_per_class for count in class_count.values()):
                    break
        if len(class_count) >= 14 and all(count >= max_images_per_class for count in class_count.values()):
            break

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def create_simple_cnn_model(input_shape, num_classes):
    """Create a simple CNN model."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_class_labels(label_to_index, filename):
    """Save class labels to a file."""
    with open(filename, 'w') as f:
        for label, index in label_to_index.items():
            f.write(f"{index}: {label}\n")
    print(f"Class labels saved to {filename}")

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.show()

def main():
    print("Starting the data loading process...")
    # Directories
    train_dir = "dataset/training"
    test_dir = "dataset/testing"

    # Load data (20 images per class)
    train_images, train_labels = load_data(train_dir, max_images_per_class=20)
    print(f"Loaded {len(train_images)} training images.")
    test_images, test_labels = load_data(test_dir, max_images_per_class=20)
    print(f"Loaded {len(test_images)} testing images.")

    # Convert labels to numeric values
    label_to_index = {label: index for index, label in enumerate(np.unique(train_labels))}
    index_to_label = {index: label for label, index in label_to_index.items()}
    train_labels = np.array([label_to_index[label] for label in train_labels])
    test_labels = np.array([label_to_index[label] for label in test_labels])

    # Save class labels to a file
    class_labels_path = "class_labels.txt"
    save_class_labels(label_to_index, class_labels_path)

    # Create model
    input_shape = (224, 224, 3)
    num_classes = len(label_to_index)
    model = create_simple_cnn_model(input_shape, num_classes)
    print("Model created.")

    # Train model
    print("Starting training process...")
    history = model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

    # Save the trained model as .h5
    model_path = "trained_model.h5"
    print("Saving the trained model as .h5...")
    model.save(model_path)

    # Evaluate the model on training data
    print("Evaluating the model on training data...")
    train_loss, train_accuracy = model.evaluate(train_images, train_labels)
    print(f"Training Accuracy: {train_accuracy}")

    # Evaluate the model on testing data
    print("Evaluating the model on testing data...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_accuracy}")

    # Predictions on testing data
    predictions = np.argmax(model.predict(test_images), axis=-1)

    # Confusion matrix and classification report
    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

    class_report = classification_report(test_labels, predictions, target_names=[index_to_label[i] for i in range(num_classes)])
    print("Classification Report:")
    print(class_report)

    # Save metrics
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Save classification report
    with open(os.path.join(metrics_dir, 'classification_report.txt'), 'w') as f:
        f.write(class_report)

    # Save confusion matrix
    cm_path = os.path.join(metrics_dir, 'confusion_matrix.txt')
    np.savetxt(cm_path, cm, fmt='%d')

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names=[index_to_label[i] for i in range(num_classes)], output_path=os.path.join(metrics_dir, 'confusion_matrix.png'))

    # Save training history
    history_path = os.path.join(metrics_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    
    # Plot training history
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(metrics_dir, 'accuracy_plot.png'))
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(metrics_dir, 'loss_plot.png'))
    plt.show()

if __name__ == '__main__':
    main()
