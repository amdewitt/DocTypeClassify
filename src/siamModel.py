
# imports

import torch
import torch.nn as nn

class SiameseModel(nn.Module):
    def __init(self):
        super(SiameseModel, self).__init__()

    def getBaseModel()
# Define Model

# Conv8 + ReLU
# LRN
# Max Pooling

# Conv4 + ReLU
# LRN
# Max Pooling

# Conv2 + ReLU
# Conv1 + ReLU
# LRN
# Max Pooling

# Dropout

# FC + ReLU + Dropout

# FC + ReLUffew s

# End Define Model

# Similarity Function ||A - B||, sum(k=1, 64, ||f(i)k - f(j)k||^2)

# Sigmoid Function 1/(1+e^-x)


# Loss = min(sum((i,j is in S),, y(i,j)log(ycarrot(i,j)) + (1 - y(i,j))log(1 - ycarrot(i,j))))