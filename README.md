# DocTypeClassify

This is a Siamese Neural Network designed to classify different medical images based on similarity.

For Training and Testing, the network requires a CSV file to point to each image in the dataset being used. The format for the CSV is ["image", "imageClass"], where image is the image path relative to the dataset directory, and class is the image's class. The latter is case-sensitive.
Each image in the set only needs to be listed once in the CSV file, along with it's class. The code will take care of the rest.

The final product will take a single image and run it over one image from each class, and return the image class with the lowest distance.

Part of the code for this project is partially based on the neural network from this article:
https://builtin.com/machine-learning/siamese-network

