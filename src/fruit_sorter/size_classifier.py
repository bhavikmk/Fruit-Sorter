import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

big_path = "Fruit-Sorter/dataset/Kesar/Big"
small_path = "Fruit-Sorter/dataset/Kesar/Small"
medium_path = "Fruit-Sorter/dataset/Kesar/Medium"

def to_binary(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img

def get_size(img):
    return img.shape[0] * img.shape[1]

def get_binary_size(img):
    return np.sum(img == 255)

# Average threshold is number of pixels in the image that are white, which signifies size of mango 

def avg_threshold_of_class(data_path):
    avg_threshold = 0
    for img_path in glob.glob(data_path + "/*.jpg"):
        img = cv2.imread(img_path)
        img = to_binary(img)
        avg_threshold += get_binary_size(img)
    return avg_threshold / len(glob.glob(data_path + "/*.jpg"))

big_avg = avg_threshold_of_class(big_path)
medium_avg = avg_threshold_of_class(medium_path)

def size_classifier(img):
    img = to_binary(img)
    if get_binary_size(img) > big_avg:
        return "Big"
    elif get_binary_size(img) > medium_avg:
        return "Medium"
    else:
        return "Small"