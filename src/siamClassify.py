# Main Driver
# Used for classification of the model
# Gets input image(s)
# Runs input image through test with each image in a pointwise dataset
# Returns the class with the lowest pairwise distance

# Imports

#import os
import tkinter as tk # Required Ultlity Modules
from tkinter import filedialog 
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import config
from siamDataset import InputComparisonDataset
from siamBaseModel import SiameseModel

def __main__():

    # Device used for computations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Print device
    # Siamese Network
    net = SiameseModel.to(device)

    print("Upload Image Files")
    root = tk.Tk() 
    root.withdraw() 
    accepted_input_types = config.accepted_input_types
    file_paths = filedialog.askopenfilenames(filetypes=accepted_input_types)
    for i in enumerate(file_paths, 0): # For each
        index, fileNameAtIndex = i[0], i[1]
        # Compare image with pointwise dataset
        print("Image {}: {}".format((index), fileNameAtIndex))

        # Testing Dataset
        classify_dataset = InputComparisonDataset(
            fileNameAtIndex,
            config.testing_csv,
            config.testing_dir,
            transform=config.transform,
        )

        # Load the dataset as pytorch tensors using dataloader
        classify_dataloader = DataLoader(
            classify_dataset,
            shuffle=True,
            num_workers=6,
            batch_size=1
        )

        best_class = ""
        best_pairwise_distance = None
        for data in enumerate(classify_dataloader, 0):
            # iterate over images, compare input to each, keep track of class with lowest distance
            inputImg, img, imgClass = data
            inputImg, img = inputImg.to(device), img.to(device)
            output0, output1 = net(inputImg, img)
            eucledian_distance = torch.mean(F.pairwise_distance(output0, output1))
            
            if(best_pairwise_distance == None or eucledian_distance < best_pairwise_distance):
                best_pairwise_distance = eucledian_distance
                best_class = imgClass
                
        # Print class with lowest Pairwise (Eucledian) Distance to input image
        print("Class of Image {}: {}".format(index, best_class))
        print("-"*20 + "\n")


# Driver Code
if __name__ == "__main__":
    __main__()