# Imports

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
#import siamUtils
import siamConfig
from siamModel import SiameseModel
from siamDataset import SiameseDataset
from siamContrastiveLoss import ContrastiveLoss

# Variables

train_dataset = SiameseDataset(
    csvFile = siamConfig.train_csv,
    directory = siamConfig.train_dir,
    transform = siamConfig.transform,
)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=siamConfig.train_shuffle,
    num_workers = siamConfig.train_num_workers,
    batch_size = siamConfig.train_batch_size
)

eval_dataset = SiameseDataset(
    csvFile = siamConfig.eval_csv,
    directory = siamConfig.eval_dir,
    transform = siamConfig.transform,
)

eval_dataloader = DataLoader(
    eval_dataset,
    shuffle = siamConfig.eval_shuffle,
    num_workers = siamConfig.eval_num_workers,
    batch_size = siamConfig.eval_batch_size
)

net = SiameseModel().to(siamConfig.device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

lossFunction = ContrastiveLoss(siamConfig.loss_margin)

# Methods

def train():
    loss = []
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = img0.to(siamConfig.device), img1.to(siamConfig.device), label.to(siamConfig.device)
        optimizer.zero_grad()
        output1, output2 = net.forward(img0, img1)
        contrastive_loss = lossFunction(output1, output2, label)
        contrastive_loss.backward()
        optimizer.step()
        loss.append(contrastive_loss.item())
    loss = np.array(loss)
    return loss.mean()/len(train_dataloader)

def eval():
    loss = []
    for i, data in enumerate(eval_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = img0.to(siamConfig.device), img1.to(siamConfig.device), label.to(siamConfig.device)
        output1, output2 = net(img0, img1)
        contrastive_loss = lossFunction(output1, output2, label)
        loss.append(contrastive_loss.item())
    loss = np.array(loss)
    return loss.mean()/len(train_dataloader)

def __main__():
    for epoch in range(0, siamConfig.epochs):
        best_eval_loss = 99999
        train_loss = train()
        eval_loss = eval()

        print(f"Epoch {(epoch+1)} of {siamConfig.epochs}")
        print(f"Training Loss: {train_loss}")
        print(f"Validation Loss: {eval_loss}")
        print("-"*20)

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            print(f"Best Validation Loss: {best_eval_loss}")
            print("-"*20)
            torch.save(net.state_dict(), siamConfig.model_path)
            print("Model Saved Successfully")

    torch.save(net.state_dict(), siamConfig.model_path)
    print("Model Saved Successfully")
      
if __name__ == "__main__":
    __main__()