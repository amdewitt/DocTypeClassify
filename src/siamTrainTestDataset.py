# Dataset class
# Manages the dataset, including training, validation, and test sets

# Class Dataset
    # Constructor(csvFile)
        # CSV File Format: Image, Class of Image
    # Get Item
    # Return Length

# Imports

import random

from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
import os
import torch


class SiameseDataset():
    # CSV File Format: CSV File Format: Image1, Image2, Label
    def __init__(self, csvFile=None, directory=None, transform=None):
        self.df = pd.read_csv(csvFile)
        self.df.columns = ["image1", "image2", "isSimilar"]
        self.dir = directory
        self.transform = transform

    # Returns results in following order: Image, Class # of Image
    def __getitem__(self, index):
        # Get image paths
        image0_path = os.path.join(self.dir, self.df.iat[index, 0])
        image1_path = os.path.join(self.dir, self.df.iat[index, 1])

        image0 = self.__pathToImage(image0_path)
        image1 = self.__pathToImage(image1_path)
        
        return (
            image0,
            image1,
            torch.from_numpy(
                np.array([int(self.train_df.iat[index, 2])], dtype=np.float32)
            )
        )

    # Path
    def __pathToImage(self, path):
        img = Image.open(path)
        img = img.convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.df)