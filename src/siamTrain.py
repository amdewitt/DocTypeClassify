# Imports

import torch
import torch.nn.functional as F
import siamUtils
import siamConfig

# Variables

#train_dataset = 

# Methods

# Loss = (1-y) * Dp^2 + y * (max(0, m - Dp))^2
# y = 1 if dissimilar, 0 otherwise
# Dp = Pairwise distance between x0 and x1
# m = loss margin for dissimilar images
def contrastive_loss(x0, x1, label, margin = siamConfig.loss_margin):
    pairwise_distance = F.pairwise_distance(x0, x1)
    margin_distance = margin - pairwise_distance
    loss = torch.mean(
        (1 - label) * torch.pow(pairwise_distance, 2) +
        label * torch.pow(torch.clamp(margin_distance, min=0.0))
    )
    return loss

def train():
    pass