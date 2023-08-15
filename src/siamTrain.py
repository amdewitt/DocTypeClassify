# Driver file for training and validating the model

# Imports
from siamDataset import PairwiseDataset # Required Network Modules
from siamBaseModel import SiameseModel
from siamLoss import ContrastiveLoss
import config # config.py, stores variables that are commonly used in model code

import torch # Required Ultlity Modules
from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
import numpy
#import matplotlib

# Variable Definitions

# Training Dataset
train_dataset = PairwiseDataset(
    config.training_csv,
    config.training_dir,
    transform=config.transform,
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
    transform=config.transform,
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
loss = ContrastiveLoss(margin = config.loss_margin)
# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

# End Variable Definitions

# Trains the model, returning average loss
def train():
    loss = []
    #counter=[]
    #iteration_number = 0
    for data in enumerate(train_dataloader, 0):
        # Get image at index
        img0, img1, class0, class1 = data
        img0, img1, = img0.to(device), img1.to(device)
        # Get Label
        label = 0
        if class0 == class1:
            label = 1
        # Initialize Optimizer
        optimizer.zero_grad()
        # Pass images through model
        output0, output1 = net(img0, img1)
        # Incur loss as needed
        contrastive_loss = loss(output0, output1, label)
        # Backpropogate to calculate model gradients
        contrastive_loss.backward()
        # Use optimizer to update model weights
        optimizer.step()
        # Add loss item to loss aray
        loss.append(contrastive_loss.item())
    # return average loss
    loss = numpy.array(loss)
    return loss.mean()/len(train_dataloader)

# Evaluates the model, returning average loss
def eval():
    loss = []
    #counter=[]
    #iteration_number = 0
    for data in enumerate(eval_dataloader, 0):
        # Get image at index
        img0, img1, class0, class1 = data
        img0, img1, = img0.to(device), img1.to(device)
        # Get Label
        label = 0
        if class0 == class1:
            label = 1
        # Pass images through model
        output0, output1 = net(img0, img1)
        # Incur loss as needed
        contrastive_loss = loss(output0, output1, label)
        # Add loss item to loss aray
        loss.append(contrastive_loss.item())
    # return average loss
    loss = numpy.array(loss)
    return loss.mean()/len(eval_dataloader)

# Main Method
# Trains the model, showing progress along the way
def __main__():
    print("Model device: " + device + "\n") # Print device
    print("-"*20 + "\n")
    for epoch in range(0, config.epochs): # Begin Training
        print("Epoch {}\n".format((epoch + 1)))
        best_eval_loss = 10000 # Initialize Best Evaluation Loss (BEL)
        train_loss = train() # Calculate Losses
        eval_loss = eval()
        print(f"Training Loss: {train_loss}\n") # Print Losses
        print(f"Validation Loss: {eval_loss}\n")
        print("-"*20 + "\n")
        if(eval_loss < best_eval_loss): # Save only if Best Evaluation Loss is surpassed
            best_eval_loss = eval_loss
            print("Best Validation Loss: {}".format(best_eval_loss))
            torch.save(net.state_dict(), config.model_path)
            print("Model Saved Successfully\n")
            print("-"*20 + "\n")
    torch.save(net.state_dict(), config.model_path) # Save again (catch-all if not saved by evaluation)
    print("Model Saved Successfully\n")

# Driver Code
if __name__ == "__main__":
    __main__()