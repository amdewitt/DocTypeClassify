# Dataset class
# Manages the dataset, including training, validation, and test sets

# Class Dataset
    # Constructor(csvFile)
        # CSV File Format: Image0, Image1, label
    # Get Item
    # Return Length

# Imports

from PIL import Image
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torchvison.transforms as transforms
from torch.utils.data.dataset import Dataset

class SiameseDataset():
    def __init__(self, csvFile=None, directory=None, transform=None)
        self.df = pd.read_csv(csvFile)
        self.df.columns = ["image1" , "image2" , "label"]
        self.dir = directory
        self.transform = transform

    def __getItem__(self, index):
        # Get image paths
        image0_path = os.path.join(self.dir, self.df.iat[index, 0])
        image1_path = os.path.join(self.dir, self.df.iat[index, 1])

        # Load images
        image0 = Image.open(image0_path)
        image1 = Image.open(image1_path)

        image0 = image0.convert("L")
        image1 = image1.convert("L")

        if self.transform is not None:
            image0 = self.transform(image0)
            image1 = self.transform(image1)
        return image0, image1, torch.from_numpy(torch.from_numpy(np.array([int(self.df.iat[index,2])],dtype=np.float32)))


    def __len__(self):
        return len(self.df)
    