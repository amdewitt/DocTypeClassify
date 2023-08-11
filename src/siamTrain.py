# Run this to train the Neural Network on the dataset

# Imports
from siamDataset import PairwiseDataset
from siamBaseModel import SiameseModel
from siamLoss import ContrastiveLoss
import config

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy

#import matplotlib

# Training Dataset
train_dataset = PairwiseDataset(
    config.training_csv,
    config.training_dir,
    transform=transforms.Compose(
        [transforms.Resize((config.height, config.width)), transforms.ToTensor()]
    ),
)

# Load the dataset as pytorch tensors using dataloader
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    num_workers=8,
    batch_size=config.batch_size
)

# Training Dataset
eval_dataset = PairwiseDataset(
    config.validation_csv,
    config.validation_dir,
    transform=transforms.Compose(
        [transforms.Resize((config.height, config.width)), transforms.ToTensor()]
    ),
)

# Load the dataset as pytorch tensors using dataloader
eval_dataloader = DataLoader(
    eval_dataset,
    shuffle=False,
    num_workers=8,
    batch_size=config.batch_size
)

# Device used for computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Print device
# Siamese Network
net = SiameseModel.to(device)
# Contrastive Loss function
loss = ContrastiveLoss(margin = config.margin)
# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

def train():
    loss = []
    #counter=[]
    #iteration_number = 0
    for data in enumerate(train_dataloader, 0):
        # Get image at index
        img0, img1, class0, class1 = data
        img0, img1, = img0.to(device), img1.to(device)
        label = 0
        if class0 == class1:
            label = 1
        optimizer.zero_grad()
        output0, output1 = net(img0, img1)
        contrastive_loss = loss(output0, output1, label)
        contrastive_loss.backward()
        optimizer.step()
        loss.append(contrastive_loss.item())
    loss = numpy.array(loss)
    return loss.mean()/len(train_dataloader)

def eval():
    loss = []
    #counter=[]
    #iteration_number = 0
    for data in enumerate(eval_dataloader, 0):
        img0, img1, class0, class1 = data
        img0, img1, = img0.to(device), img1.to(device)
        label = 0
        if class0 == class1:
            label = 1
        output0, output1 = net(img0, img1)
        contrastive_loss = loss(output0, output1, label)
        loss.append(contrastive_loss.item())
    loss = numpy.array(loss)
    return loss.mean()/len(eval_dataloader)

# Driver Code
# Trains the model, showing progress along the way
def __main__():
    print("Model device: " + device + "\n") # Print device
    print("-"*20 + "\n")
    for epoch in range(0, config.epochs): # Begin Training
        print("Epoch {}\n".format((epoch + 1)))
        best_eval_loss = 10000
        train_loss = train()
        eval_loss = eval()
        print(f"Training Loss: {train_loss}\n") # Print Losses
        print(f"Validation Loss: {eval_loss}\n")
        print("-"*20 + "\n")
        if(eval_loss < best_eval_loss):
            best_eval_loss = eval_loss
            print("Best Validation Loss: {}".format(best_eval_loss))
            torch.save(net.state_dict(), "/savedModels/model.pth")
            print("Model Saved Successfully")
            print("-"*20 + "\n")

# Driver Code
if __name__ == "__main__":
    __main__()