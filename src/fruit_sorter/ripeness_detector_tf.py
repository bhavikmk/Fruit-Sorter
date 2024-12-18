import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import time

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = 'models/mango_ripeness_mobilenetv2.h5'
TEST_IMAGE_PATH = 'dataset/Stage_3 (Ripe)/Test/IMG20200717083707.jpg'
BASE_DIR = 'dataset/'

# Load pre-trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


CLASS_LABELS = ['Stage_0 (Unripe)', 'Stage_1 (Early Ripe)', 'Stage_2 (Partially Ripe)', 'Stage_3 (Ripe)']

init = time.time()

image_array = img_to_array(load_img(TEST_IMAGE_PATH, target_size=IMG_SIZE)) / 255.0
prediction = model.predict(np.expand_dims(image_array, axis=0))
predicted_class = CLASS_LABELS[np.argmax(prediction)]

finish = time.time()

print(f"Predicted Class: {predicted_class}, time = {(finish - init)*1000} ms")
