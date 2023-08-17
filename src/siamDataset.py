import os
import pandas
import math
from PIL import Image

# Pairwise dataset used for training and testing
# CSV Format: Image, Class
class SiameseDataset():
    def __init__(self, csvFile = None, directory = None, transform = None):
        self.df = pandas.read_csv(csvFile) # Read CSV File
        self.df.columns = ["ImagePath", "ImageClass"] # Initialize Columns
        self.dir = directory
        self.transform = transform

    def __getItem__(self, index):
        if index < 0 or index > self.__len__():
            index0, index1 = 0,0
        else:
            index0, index1 = self.__indexToTriMatrixCoords(index)
        img0 = self.__pathToImage(self.df.iat[index0, 0], isRelativePath=True)
        img1 = self.__pathToImage(self.df.iat[index1, 0], isRelativePath=True)

        class0 = str(self.df.iat[index0, 1])
        class1 = str(self.df.iat[index1, 1])

        label = 0
        if class0 == class1:
            label = 1

        return img0, img1, label

    # Helper Method, converts path to image
    def __pathToImage(self, imgPath=None, isRelativePath=True):
        if isRelativePath is not True:
            path = str(imgPath)
        else:
            path = str(os.path.join(self.dir, imgPath))
        
        img = Image.open(path)
        img = img.convert("L")

        if self.transform is not None:
            img = self.transform(img)
        return img
        
    # Gets distinct pair of indices from single index
    def __indexToTriMatrixCoords(self, index):
        i = math.ceil(math.sqrt(2 * (index + 1) + 0.25) - 0.5) # Upper Index
        j = index - (i - 1) * i / 2 # Lower Index
        return i, j
        
    def __len__(self):
        n = len(self.df)
        return n * (n - 1) / 2
        
