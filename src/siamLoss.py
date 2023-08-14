# Contrastive Loss
# Pairwise Distance-based Loss for comparing pairs of images
# L(W(X1, X2, Y)) = y * Dp^2 + (1-y) * 0.5 * max((margin - Dp)^2, 0)
# Y: 1 if images are of same class, 0 otherwise
# Dp: Pairwise distance (aka Eucledian distance)
# margin: loss margin if images are of different classes

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
        lossSimilar = y * torch.pow(euclidean_distance, 2) # loss if two items are similar
        lossDissimilar = (1 - y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0)) # loss if two items are dissimilar
        loss = torch.mean(lossSimilar + lossDissimilar)
        return loss