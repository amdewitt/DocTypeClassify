# DocTypeClassify

This is a Siamese Neural Network designed to classify different medical images based on similarity.

For Training and Testing, the network requires a CSV file to point to each image in the dataset being used. The format for the CSV is ["image", "imageClass"], where image is the image path relative to the dataset directory, and class is the image's class (case-insensitive).
Each image in the set only needs to be listed once in the CSV file, along with it's class. The code will take care of the rest.

The final product will take a an image and make pairs with it and each image in the test directory. These pairs will be put through the model, which will return the image class with the lowest distance.

    For demonstration purposes, a mock dataset has been added, along with the corresponding CSVs.

The config.py file lets you set customize certain parameters without needing to change the rest of the code.

This project is partially based on the neural network from this article:
https://builtin.com/machine-learning/siamese-network

