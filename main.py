import numpy as np
import tensorflow as tf
from PIL import Image

def load_image(image_path, input_shape):
    """Load and preprocess the image."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def main(model_path, image_path, class_labels):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare the input data.
    input_shape = input_details[0]['shape']
    input_data = load_image(image_path, input_shape)

    # Set the input tensor.
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference.
    interpreter.invoke()

    # Get the output tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the predicted class with the highest probability
    predicted_class = np.argmax(output_data)
    predicted_label = class_labels[predicted_class]

    print("Output probabilities:", output_data)
    print("Predicted class:", predicted_class)
    print("Predicted label:", predicted_label)

if __name__ == '__main__':
    model_path = 'ei-clothes_type_3_colors-transfer-learning-tensorflow-lite-float32-model (1).lite'
    image_path = 'test/image1.jpg'
    class_labels = [
        'black_dress', 'black_pants', 'black_shirt', 'black_shoes', 'black_shorts',
        'blue_dress', 'blue_pants', 'blue_shirts', 'blue_shorts', 'red_dress',
        'red_pants', 'red_shoes', 'white_dress', 'white_pants','white_shorts',
    ]

    main(model_path, image_path, class_labels)
