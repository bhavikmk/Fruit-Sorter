# import the required libraries
import glob
from logging import root
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# define path to dataset and create a dataframe

big_path = "Fruit-Sorter/dataset/Kesar/Big"
small_path = "Fruit-Sorter/dataset/Kesar/Small"
medium_path = "Fruit-Sorter/dataset/Kesar/Medium"

class Dataset(Dataset):

    def __init__(self, label ,root, transform=None):

        self.label = label
        self.root = root
        self.transform = transform

    def __len__(self):
        return len([name for name in os.listdir(root) if os.path.isfile(os.path.join(root, name))])

    def __getitem__(self, index):

        img_name = os.path.join(root, os.listdir(root)[index])
        image = mpimg.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image / 255.0
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        return image, self.label
