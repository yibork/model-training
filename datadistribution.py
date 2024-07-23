import os
from collections import defaultdict

def count_images_per_class(dataset_dir, subset='training'):
    # Dictionary to hold the count of images per class
    class_counts = defaultdict(int)
    total_images = 0

    # Directory path for the specified subset (training or testing)
    subset_dir = os.path.join(dataset_dir, subset)

    # Iterate over all files in the subset directory
    for root, _, files in os.walk(subset_dir):
        for file in files:
            # Get the class name from the filename
            class_name = file.split('.')[0]  # Assumes class name is before the first period
            class_counts[class_name] += 1
            total_images += 1

    return class_counts, total_images

# Example usage
dataset_directory = 'dataset/'  # Replace with the path to your dataset
training_class_image_counts, training_total_images = count_images_per_class(dataset_directory, subset='training')
testing_class_image_counts, testing_total_images = count_images_per_class(dataset_directory, subset='testing')

# Print the results for training dataset
print("Training Dataset:")
for class_name, count in training_class_image_counts.items():
    print(f"Class: {class_name}, Number of images: {count}")
print(f"Total number of images: {training_total_images}")

# Print the results for testing dataset
print("\nTesting Dataset:")
for class_name, count in testing_class_image_counts.items():
    print(f"Class: {class_name}, Number of images: {count}")
print(f"Total number of images: {testing_total_images}")
