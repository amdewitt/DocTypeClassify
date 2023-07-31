# Dataset class
# Manages the dataset, including training, validation, and test sets

# Class Dataset
    # Constructor(csvFile)
        # CSV File Format: Image0, Image1, label
    # Get Item
    # Return Length

# Imports

from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
import os

import torch


class SiameseDataset():
    # CSV File Format: Image 1, Class of Image 1, Image 2, Class of Image 2
    def __init__(self, csvFile=None, directory=None, transform=None):
        self.df = pd.read_csv(csvFile)
        self.df.columns = ["image0" , "class0", "image1" , "class1"]
        self.dir = directory
        self.transform = transform

    # Returns results in following order: Image 1, Image 2, Class of Image 1, Class of Image 2, Equality of the Classes
    def __getItem__(self, index):
        # Get image paths
        image0_path = os.path.join(self.dir, self.df.iat[index, 0])
        image1_path = os.path.join(self.dir, self.df.iat[index, 2])

        # Load images
        image0 = Image.open(image0_path)
        image1 = Image.open(image1_path)

        image0 = image0.convert("L")
        image1 = image1.convert("L")

        # Transform Images
        if self.transform is not None:
            image0 = self.transform(image0)
            image1 = self.transform(image1)

        # Get and Compare Image Classes

        class0 = self.df.iat[index,1]
        class1 = self.df.iat[index,3]

        # Prepare Labels for Images
        isSameClass = 0
        if class0.__eq__(class1):
            isSameClass = 1



        return image0, image1, class0, class1, isSameClass


    def __len__(self):
        return len(self.df)
    