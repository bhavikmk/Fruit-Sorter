import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

import os
import matplotlib.pyplot as plt

base_dir = 'dataset/'

image_path = "dataset/Stage_0 (Unripe)/Training/IMG20200713141551.jpg"  

# Define image dimensions and batch size
IMG_SIZE = (224, 224)  # Required input size for MobileNetV2
BATCH_SIZE = 32

model = tf.keras.models.load_model('models/mango_ripeness_mobilenetv2.h5')

from tensorflow.keras.preprocessing.image import load_img, img_to_array

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80-20 split for train-validation
)


train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Path to the test image
test_image_path = '/kaggle/input/ripeness-detection-of-mango/Stage_0 (Unripe)/Test/IMG20200713142258.jpg'

# Preprocess the test image
image = load_img(test_image_path, target_size=IMG_SIZE)
image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict ripeness stage
prediction = model.predict(image_array)
predicted_class = np.argmax(prediction)
class_labels = list(train_generator.class_indices.keys())

print(f"Predicted Class: {class_labels[predicted_class]}")
