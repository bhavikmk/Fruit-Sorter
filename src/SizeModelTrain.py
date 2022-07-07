import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

"""### Importing dataset Google drive"""

from google.colab import drive
drive.mount('/content/drive')

img = plt.imread(
    "/content/drive/MyDrive/Colab Notebooks/Fruit Sorter data/Kesar/Big/D116.jpg")
plt.imshow(img)

"""### Size : 3 Mango Categories 

* Big
* Medium
* Small

## Organization and variables of code

* img_path : Dataset path
* image_count : Total images in dataset
* small, medium, large : Path List of all images im respective categories
"""

img_path = '/content/drive/MyDrive/Colab Notebooks/Fruit Sorter data/Kesar'
img_path = pathlib.Path(img_path)
image_count = len(list(img_path.glob('*/*.jpg')))

print(image_count)

small = list(img_path.glob('Small/*'))
medium = list(img_path.glob('Medium/*'))
large = list(img_path.glob('Large/*'))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    img_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    img_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    
class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

num_classes = 3

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

model.summary()

model.save('mango_size_detector.h5')
