# Handles the training of the model
# Train
#   Main
#   Train

# Imports

from siamDataset import SiameseDataset
from siamBaseModel import SiameseModel
from contrastiveLoss import ContrastiveLoss

import torch as th
from torch.utils.data import DataLoader

# Load the the dataset from raw image folders
training_dataset = SiameseDataset(training_csv,training_dir, transform=transforms.Compose([transforms.Resize((105,105)), transforms.ToTensor()]))

trainiing_dataloader = DataLoader(training_dataset)

# Declare Siamese Network
net = SiameseModel().cuda()
# Declare Loss Function
criterion = ContrastiveLoss()
# Declare Optimizer
optimizer = th.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

# Declare Siamese Network
net = SiameseModel().cuda()
# Decalre Loss Function
criterion = ContrastiveLoss()

# Training Method
# Iterate over all distinct pairs of different images for each epoch
# for(int c = 0; i < epochs; ++c)
#  for(int i = 0; i < __len__(); ++i)
#  img0 = img(i), class0 = imgClass(i)
#   for(int j = i + 1; j < __len__(); ++j)
#   img1 = img(j), class1 = imgClass(j)
#   label = sameClass(i, j)

"""def train():
    loss=[]
    counter=[]
    iteration_number = 0
    for epoch in range(1,config.epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()   
        print("Epoch {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
    show_plot(counter, loss)  
    return net"""