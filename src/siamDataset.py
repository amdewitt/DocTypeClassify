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

import siamClassDict

#import torch


class SiameseDataset():
    # CSV File Format: CSV File Format: Image, Class of Image
    def __init__(self, csvFile=None, directory=None, transform=None):
        self.df = pd.read_csv(csvFile)
        self.df.columns = ["image", "class"]
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

        # Get and Compare Image Clsses
        class0 = str(siamClassDict.__findClass__(self.df.iat[index,1]))

        return image0, class0
    
    def __getItemPair__(self, index0, index1):
        image0, class0 = self.__getSingleItem__(index0)
        image1, class1 = self.__getSingleItem__(index1)
        sameClass = 0
        if class0 == class1:
            sameClass = 1
        return image0, image1, sameClass
    
    def __getRandomItemFromClass__(self, targetClass):
        imgList = []
        nImagesInClass = 0
        for i in self.df:
            classQuery = siamClassDict.__findClass__(targetClass)
            imgClass = siamClassDict.__findClass__(self.df.iat[i,1])
            if classQuery == imgClass:
                classImg = [self.df.iat[i,0]]
                imgList.extend[classImg]
                nImagesInClass += 1
            else:
                continue
        
        r = random.randint(0, (nImagesInClass - 1))
        randImg = imgList[r]
        return randImg
        
        #return randImg, imgClass

    def __len__(self):
        return len(self.df)