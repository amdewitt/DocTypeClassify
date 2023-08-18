import os
from torch.utils.data import Dataset
import pandas
import math

import siamUtils

# Pairwise dataset used for training and testing
# CSV Format: Image, Class
class SiameseDataset(Dataset):
    def __init__(self, csvFile = None, directory = None, transform = None):
        super(SiameseDataset, self).__init__()
        self.df = pandas.read_csv(csvFile) # Read CSV File
        self.df.columns = ["ImagePath", "ImageClass"] # Initialize Columns
        self.dir = directory
        self.transform = transform

    def __getitem__(self, index):
        if index < 0 or index > self.__len__():
            index0, index1 = 0,0
        else:
            index0, index1 = self.__indexToTriMatrixCoords(index)
        img0 = siamUtils.imagePathToImage(os.path.join(self.dir, self.df.iat[index0, 0]), transform = self.transform)
        img1 = siamUtils.imagePathToImage(os.path.join(self.dir, self.df.iat[index1, 0]), transform = self.transform)

        class0 = str(self.df.iat[index0, 1])
        class1 = str(self.df.iat[index1, 1])

        return img0, img1, class0, class1
        
    def __len__(self):
        n = len(self.df)
        return int(n * (n - 1) / 2)
    
    # Gets distinct pair of indices from single index
    def __indexToTriMatrixCoords(self, index):
        i = int(math.ceil(math.sqrt(2 * int(index + 1) + 0.25) - 0.5)) # Upper Index
        j = int(int(index) - (i - 1) * i / 2) # Lower Index
        return i, j
    
# Used for image classification with an input image
# CSV Format: Relative Image Path, Image Class
class ClassificationDataset:
    def __init__(self, inputPath = None, csvFile = None, directory = None, transform = None):
        #super(ClassificationDataset, self).__init__()
        self.input_image_path = inputPath
        self.df = pandas.read_csv(csvFile) # Read CSV File
        self.df.columns = ["ImagePath", "ImageClass"] # Initialize Columns
        self.dir = directory
        self.transform = transform

    # Gets the input image and an image from the CSV file
    def __getitem__(self, index):
        inputImg = siamUtils.imagePathToImage(self.input_image_path, transform = self.transform, rootDir = None)
        img = siamUtils.imagePathToImage(self.df.iat[index, 0], transform = self.transform, rootDir = self.dir)
        imgClass = str(self.df.iat(index, 1))
        return inputImg, img, imgClass

    def __len__(self):
        return len(self.df)