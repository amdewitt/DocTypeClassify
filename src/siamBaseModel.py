# The base architecture of the model
# Class SiameseBaseModel
    # Constructor
    # Forward - puts two images through model
    # Output - Class input is closest to

# Imports
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import  StepLR

class SiameseModel(nn.Module):
    # Constructor
    def __init__(self)
        super(SiameseModel, self).__init__()
        
        # use resnet for
        self.resnet = torchvision.models.resnet18(pretrained=False)

        self.