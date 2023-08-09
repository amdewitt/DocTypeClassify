
# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Nodule):
    def __init__(self, margin = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    # L(W, (Y, X1, X2)i) = y*Ls*D^i_W + ()1-y)*Ld*D^i_W
    # D^i_W = euclidean distance b/w two points
    # Y = 1 if similar, 0 if dissimilar
    # X1, X2 = two datapoints
    def forward(self, x0, x1, y):
        euclidean_distance = F.pairwise_distance(x0, x1)
        lossSimilar = y * torch.pow(euclidean_distance, 2) # loss if two items are similar
        lossDissimilar = (1 - y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0))
        loss = torch.mean(lossSimilar + lossDissimilar)
        return loss