#######################################

####### Overview #######

This project is a Siamese Neural Network, designed to determine if two images are similar.

The datasets use three CSVs and three separate datasets, one for training, one for validation, and one for testing. The path to each is stored in the siamConfig.py file, along with their respective rood directories.

The CSVs point to the images being used, and are to be written in the following format:

relativeImagepath0,imageClass0
relativeImagePath1,imageClass1
relativeImagePath2,imageClass2
...
relativeImagePathN,imageClassN

The first row may or may not be treated as a header, depending on the value of the treat_first_line_as_header variable stored in siamConfig.py.

relativeImagePathN is an image path with the file extension (.xxx, such as .png), relative to the root directory as specified in the siamConfig.py file.
imageClassN is the class that relativeImagePathN's image belong's to. The class is case-insensitive (class = CLASS = ClAsS).

The code iterates over each distinct pair of images during training, validation, and testing. When classifying an input image, the code iterates over each pair involving the input image and an image in the testing CSV file.

The siamConfig.py file also allows for easy customization of other variables commonly used in the model code, such as the number of training epochs.

Part of the code is partly based on code from this article: https://builtin.com/machine-learning/siamese-network

#######################################

####### Using the Main Drivers #######

####### siamCompare.py #######

Takes an existing directory and a CSV file, in that order. Both are prompted by their respective dialogs.

The CSV file should be written in the following format:

Header
relativeImagePath0
relativeImagePath1
relativeImagePath2
...
relativeImagePathN

After accepting both inputs, siamCompare.py will iterate over each distinct pair, outputing the image paths, the (predicted) pairwise distance b/w the images, and the label for each.

####### siamClassify.py #######

Takesa a set of input image, prompted by a dialog.

For each image in the set, siamClassify.py pairs the input with each image in the testing CSV file, and returns the class with the lowest distance.