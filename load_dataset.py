import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(root_dir):
    images = []
    labels = []
    label_map = {}
    
    # Create mapping for class labels
    classes = sorted(os.listdir(root_dir))
    for idx, class_name in enumerate(classes):
        label_map[class_name] = idx
    
    # Load images and labels
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to consistent dimensions
                    img = cv2.resize(img, (28, 28))
                    # Normalize pixel values
                    img = img / 255.0
                    images.append(img)
                    labels.append(label_map[class_name])
    
    return np.array(images), np.array(labels), label_map

# Load training data
X_train, y_train, label_map = load_dataset('handwritten-english-characters-and-digits/combined_folder/train')

# Load test data
X_test, y_test, _ = load_dataset('handwritten-english-characters-and-digits/combined_folder/test')

# Reshape for CNN input (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
