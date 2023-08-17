# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # Loss 
    def forward(self, x0, x1, y):
        euclid_distance = F.pairwise_distance(x0, x1)
        margin_distance = self.margin - euclid_distance
        loss = torch.mean((1-y)*torch.pow(euclid_distance, 2) + y*torch.pow(torch.clamp(margin_distance, min = 0.0), 2))
        return loss