import os
from PIL import Image
import pandas
import torch
import numpy
import math

import siamConfig
import siamUtils
    
# Used for image classification with an input image
# CSV Format: Relative Image Path, Image Class
class SiameseDataset:
    def __init__(self, csvFile = None, directory = None, transform = None):
        if siamConfig.treat_first_line_as_header == False:
            self.df = pandas.read_csv(csvFile, header=None) # Read CSV File (headers not included in file)
        else:
            self.df = pandas.read_csv(csvFile)
        self.df.columns = ["ImagePath", "imageClass"] # Initialize Columns
        self.dir = directory
        self.transform = transform

    def __indexToTriMatrixCoords(self, index):
        if index < 0 or index >= self.__len__() or self.__len__() < 2:
            return 0, 0
        # Optimized formulas for index conversion
        i = int(math.ceil(math.sqrt(2 * int(index + 1) + 0.25) - 0.5)) # Upper Index
        j = int(int(index) - (i - 1) * i / 2) # Lower Index
        return i, j

    # Gets the input image and an image from the CSV file
    def __getitem__(self, index):
        index0, index1 = self.__indexToTriMatrixCoords(index)

        img0 = siamUtils.imagePathToImage(os.path.join(self.dir, self.df.iat[index0, 0]))
        img1 = siamUtils.imagePathToImage(os.path.join(self.dir, self.df.iat[index1, 0]))

        class0 = str(self.df.iat[index0, 1]).lower()
        class1 = str(self.df.iat[index1, 1]).lower()

        if class0 == class1:
            diffClasses = 0
        else:
            diffClasses = 1

        return img0, img1, torch.from_numpy(numpy.array([int(diffClasses)], dtype=numpy.float32)), class0, class1
    
    # Get item paths
    def __getItemPaths__(self, index):
        index0, index1 = self.__indexToTriMatrixCoords(index)

        img0 = str(os.path.join(self.dir, self.df.iat[index0, 0]))
        img1 = str(os.path.join(self.dir, self.df.iat[index1, 0]))

        return img0, img1

    def __len__(self):
        return int((len(self.df) * (len(self.df) - 1)) / 2)
    
####################

class PairwiseDataset:
    def __init__(self, csvFile = None, directory = None, transform = None):
        self.df = pandas.read_csv(csvFile) # Read CSV File (headers always included in file)
        self.df.columns = ["ImagePath"] # Initialize Columns
        self.dir = directory
        self.transform = transform

    def __indexToTriMatrixCoords(self, index):
        if index < 0 or index >= self.__len__() or self.__len__() < 2:
            return 0, 0
        # Optimized formulas for index conversion
        i = int(math.ceil(math.sqrt(2 * int(index + 1) + 0.25) - 0.5)) # Upper Index
        j = int(int(index) - (i - 1) * i / 2) # Lower Index
        return i, j

    # Gets the input image and an image from the CSV file
    def __getitem__(self, index):
        index0, index1 = self.__indexToTriMatrixCoords(index)

        img0 = siamUtils.imagePathToImage(os.path.join(self.dir, self.df.iat[index0, 0]))
        img1 = siamUtils.imagePathToImage(os.path.join(self.dir, self.df.iat[index1, 0]))

        return img0, img1
    
    # Get item paths
    def __getItemPaths__(self, index):
        index0, index1 = self.__indexToTriMatrixCoords(index)

        img0 = str(os.path.join(self.dir, self.df.iat[index0, 0]))
        img1 = str(os.path.join(self.dir, self.df.iat[index1, 0]))

        return img0, img1

    def __len__(self):
        return int((len(self.df) * (len(self.df) - 1)) / 2)