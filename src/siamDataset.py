# Dataset class
# Manages the dataset, including training, validation, and test sets

# Class Dataset
    # Constructor(csvFile)
        # CSV File Format: Image, Class of Image
    # Get Item
    # Return Length

# Imports

from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
import os

#import torch


class SiameseDataset():
    # CSV File Format: CSV File Format: Image, Class of Image
    def __init__(self, csvFile=None, directory=None, transform=None):
        self.df = pd.read_csv(csvFile)
        self.df.columns = ["image0" , "image1" , "class0" , "class1"]
        self.dir = directory
        self.transform = transform

    # Returns results in following order: Image, Class # of Image
    def __getSingleItem__(self, index):
        # Get image paths
        image0_path = os.path.join(self.dir, self.df.iat[index, 0])

        # Load images
        image0 = Image.open(image0_path)

        image0 = image0.convert("L")

        # Transform Images
        if self.transform is not None:
            image0 = self.transform(image0)

        # Get and Compare Image Classes

        class0 = self.df.iat[index,1]

        return image0, class0

    def __len__(self):
        return len(self.df)