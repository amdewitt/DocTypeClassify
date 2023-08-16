# Dataset class
# Manages the dataset, including training, validation, and test sets

# Class Dataset
    # Constructor(csvFile)
        # CSV File Format: Image, Class of Image
    # Get Item
    # Return Length

# Imports

from PIL import Image
import pandas as pd
import os
import math

# (Helper) Takes an image path and converts it to a common image format
def pathToImage(img_path, transform):
    img = Image.open(img_path)
    # convert images to common format
    img = img.convert("L")
    # transform image
    if transform is not None:
        img = transform(img)
    return img

# Class used for Training and Testing
class PairwiseDataset():
    # CSV File Format: CSV File Format: Image, Class
    def __init__(self, csvFile=None, directory=None, transform=None):
        # Prepare csv file for reading
        self.df = pd.read_csv(csvFile)
        self.df.columns = ["image", "imageClass"]
        # Set other necessary variables
        self.dir = directory
        self.transform = transform

    def __getitem__(self, index): 
        # Overflow code
        if(index < 0 or index >= self.__len__()):
            index0, index1 = self.__indexToTriMatrixCoords(-1)
        else:
            # Get indices of image pair
            index0, index1 = self.__indexToTriMatrixCoords(index)
        # Get images
        #img0 = self.__pathToImage(self.df.iat[index0, 0])
        #img1 = self.__pathToImage(self.df.iat[index1, 0])
        img0 = pathToImage(os.path.join(self.dir, self.df.iat[index0, 0]), self.transform)
        img1 = pathToImage(os.path.join(self.dir, self.df.iat[index1, 0]), self.transform)
        # Get Classes
        class0 = str(self.df.iat[index0, 1]).lower()
        class1 = str(self.df.iat[index0, 1]).lower()
        return img0, img1, class0, class1

    # (Helper) Turns a 1d index into a pair of distinct coordinates to fetch a pair of images from the CSV file
    def __indexToTriMatrixCoords(self, index):
        # default if index is not in range
        if(index < 0 or index >= self.__len__()):
            return 1, 0
        # Optimized formulas for getting triangular matrix coordinates
        # (sourced from https://stackoverflow.com/questions/40950460/how-to-convert-triangular-matrix-indexes-in-to-row-column-coordinates)
        i = int(math.ceil(math.sqrt(2 * (index + 1) + 0.25) - 0.5))
        j = int((index + 1) - (i - 1) * i / 2 - 1)
        return i, j
    
    # Gets the total number of distinct pairs that can be fetched from the csv file
    def __len__(self):
        return int(len(self.df) * (len(self.df) - 1) / 2)

# Class used for classification with an input
class InputComparisonDataset():
    # CSV File Format: CSV File Format: Image, Class
    def __init__(self, inputImagePath=None, csvFile=None, directory=None, transform=None):
        # Prepare csv file for reading
        self.inputImagePath = inputImagePath
        self.df = pd.read_csv(csvFile)
        self.df.columns = ["image", "imageClass"]
        # Set other necessary variables
        self.dir = directory
        self.transform = transform

    def __getitem__(self, index):
        # Overflow code
        if(index < 0 or index >= self.__len__()):
            index = 0
        # Get input Image
        inputImg = pathToImage(self.inputImagePath, self.transform)
        # Get image
        img = pathToImage(os.path.join(self.dir, self.df.iat[index, 0]), self.transform)
        # Get Classes
        imgClass = str(self.df.iat[index, 1]).lower()
        return inputImg, img, imgClass
    
    def __len__(self):
        return len(self.df)