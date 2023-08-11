# Run this to train the Neural Network on the dataset

# Imports
from siamDataset import SiameseDataset
from siamBaseModel import SiameseModel
from siamLoss import ContrastiveLoss
import config

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy

#import matplotlib

# Training Dataset
train_dataset = SiameseDataset(
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

# Siamese Network
net = SiameseModel.cuda()
# Contrastive Loss function
loss = ContrastiveLoss()
# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

# Driver Code
# Trains the model, showing progress along the way
def __main__():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Model device: " + device + "\n--------------------\n")
    for epoch in range(0, config.epochs):
        print("Epoch {}".format((epoch + 1)))
        train_loss = train()
        print(f"Training Loss {train_loss}\n")
        print("--------------------\n")
    torch.save(net.state_dict(), "/trainedModel/model.pth")
    print("Model Saved Successfully")
    

def train():
    loss = []
    #counter=[]
    #iteration_number = 0
    for data in enumerate(train_dataloader, 0):
        # Get image at index
        img0, img1, class0, class1 = data
        img0, img1, = img0.cuda(), img1.cuda()
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

# Driver Code
if __name__ == "__main__":
    __main__()