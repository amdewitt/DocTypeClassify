# Imports
from siamDataset import SiameseDataset
from siamBaseModel import SiameseModel
from siamLoss import ContrastiveLoss
import config

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#import matplotlib

# Dataset
siamese_dataset = SiameseDataset(
    config.training_csv,
    config.training_dir,
    transform=transforms.Compose(
        [transforms.Resize((105, 105)), transforms.ToTensor()]
    ),
)

# Load the dataset as pytorch tensors using dataloader
train_dataloader = DataLoader(
    siamese_dataset,
    shuffle=True,
    num_workers=8,
    batch_size=config.batch_size) 
# Siamese Network
net = SiameseModel.cuda()
# Contrastive Loss function
loss = ContrastiveLoss()
# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

def __main__():
    device = torch.device('cuda' if torch.cuda.is_available() else ' cpu')
    print("Model device: " + device + "\n------------------\n")
    model = train()
    torch.save(model.state_dict(), "model.pt")
    print("Model Saved Successfully:\n")
    

def train():
    loss = []
    counter = []
    iteration_number = 0
    for epoch in range(0, config.epochs):
        for data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            contrastive_loss = loss(output1, output2, label)
            contrastive_loss.backward()
            optimizer.step()
        print("Epoch: {}, Current Loss: {}\n".format(epoch, loss))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(contrastive_loss.item())
    #show_plot(counter, loss)
    return net