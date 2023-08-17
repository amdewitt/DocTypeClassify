# Imports

#import torch
import torch.nn as nn

class SiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.convolutionalLayer = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )

        self.fullyConnectedLayer = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
           
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
           
            nn.Linear(128,2),
            #nn.Linear(2, 1),
            #nn.Sigmoid(),
        )
        
    def forward_once(self, x):
        out = self.convolutionalLayer(x)
        out = out.view(out.size()[0], -1)
        out = self.fullyConnectedLayer(out)
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2