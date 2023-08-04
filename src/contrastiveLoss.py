
# Imports

import torch
import torch.nn as nn

class ContrastiveLoss(nn.Nodule):
    def __init__(self, margin = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, x0, x1, isFromSameClass):
        y = 0
        if (isFromSameClass == 1 or isFromSameClass == True):
            y = 1

        diff = x0 - x1
        euclideanDist_sq = torch.pow(diff, 2)
        euclideanDist = torch.sqrt(euclideanDist_sq)

        marginDiff = self.margin - euclideanDist
        marginDist = torch.clamp(marginDiff, min=0.0)

        loss = y*euclideanDist_sq + (1-y)*torch.pow(marginDist, 2)
        contrastiveLoss = torch.sum(loss) / 2.0 / x0.size()[0]
        return contrastiveLoss