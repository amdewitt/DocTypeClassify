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
import torch as th

import siamClassDict

#import torch


class SiameseDataset():
    # CSV File Format: CSV File Format: Image1, Image2, IsSameClass
    def __init__(self, csvFile=None, directory=None, transform=None):
        self.df = pd.read_csv(csvFile)
        self.df.columns = ["image1", "image2", "sameclass"]
        self.dir = directory
        self.transform = transform

    # Returns results in following order: Image, Class # of Image
    def __getitem__(self, index):
        # getting the image path
        image0_path=os.path.join(self.train_dir,self.train_df.iat[index,0])
        image1_path=os.path.join(self.train_dir,self.train_df.iat[index,1])
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1 , th.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
    
    """def __getRandomItemFromClass__(self, targetClass):
        imgList = []
        nImagesInClass = 0 # number of images in the target class
        for i in self.df:
            classQuery = siamClassDict.__findClass__(targetClass)
            imgPath, imgClass = self.__getSingleItem__(i)
            if classQuery == imgClass:
                classImg = [imgPath]
                imgList.extend[classImg]
                nImagesInClass += 1
            else:
                continue
        
        # Gets random index and returns image at that index
        r = random.randint(0, (nImagesInClass - 1))
        randImg = imgList[r]
        return randImg"""
        
        #return randImg, imgClass

    def __len__(self):
        return len(self.df)