# Driver file for testing the model

# Imports

from siamDataset import PairwiseDataset # Required Network Modules
from siamBaseModel import SiameseModel
import config # config.py, stores variables that are commonly used in model code

import torch # Required Ultlity Modules
from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
import torch.nn.functional as F

# Testing Dataset
test_dataset = PairwiseDataset(
    config.testing_csv,
    config.testing_dir,
    transform=config.transform,
)

# Load the dataset as pytorch tensors using dataloader
test_dataloader = DataLoader(
    test_dataset,
    shuffle=True,
    num_workers=6,
    batch_size=1
)


# Device used for computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Print device
# Siamese Network
net = SiameseModel.to(device)

# Tests the model's accuracy
def test():
    count = 0
    for data in enumerate(test_dataloader, 0):
        img0, img1, class0, class1 = data # Get Images
        img0, img1 = img0.to(device), img1.to(device) # Use device for images
        label = "Different Classes" # Get Label
        if class0 == class1:
            label = "Same Classes"
        output0, output1 = net(img0, img1) # Get Embeddings
        eucledian_distance = F.pairwise_distance(output0, output1) # Get Pairwise Distance
        print("Class of Image 0: {}, Class of Image 1: {}, Label: {}".format(class0, class1, label)) # Print Results
        print("Predicted Eucledian Distance: {}\n".format(eucledian_distance.item()))
        print("-"*20 + "\n")
        count += 1
        if count >= config.max_tests: # Don't go past max tets
            break

# Main Method
# Tests the model accuracy
def __main__():
    net.load_state_dict(torch.load(config.model_path))
    test()

# Driver Code
if __name__ == "__main__":
    __main__()