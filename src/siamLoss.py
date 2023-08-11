
# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Nodule):
    def __init__(self, margin = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin # loss margin
    
    # Loss Function
    def forward(self, x0, x1, y):
        euclidean_distance = F.pairwise_distance(x0, x1)
        lossSimilar = y * torch.pow(euclidean_distance, 2) / 2 # loss if two items are similar
        lossDissimilar = (1 - y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0)) / 2 # loss if two items are dissimilar
        loss = torch.mean(lossSimilar + lossDissimilar)
        return loss