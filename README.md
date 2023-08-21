This project is a Siamese Neural Network, designed to determine if two images are similar.

The datasets use three CSVs and three separate datasets, one for training, one for validation, and one for testing. The path to each is stored in the siamConfig.py file, along with their respective rood directories.

The CSVs point to the images being used, and are to be written in the following format:

relativeImagepath0,imageClass0
relativeImagePath1,imageClass1
relativeImagePath2,imageClass2
...

relativeImagePathN is an image path with the file extension (.xxx, such as .png), relative to the root directory as specified in the siamConfig.py file.
imageClassN is the class that relativeImagePathN's image belong's to. The class is case-insensitive (class = CLASS = ClAsS).

The code iterates over each distinct pair of images during training, validation, and testing. When classifying an input image, the code iterates over each pair involving the input image and an image in the testing CSV file.

The siamConfig.py file also allows for easy customization of other variables commonly used in the model code, such as the number of training epochs.

Part of the code is partly based on code from this article: https://builtin.com/machine-learning/siamese-network