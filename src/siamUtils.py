# Imports

import torch
import torchvision.transforms as transforms

import numpy
import matplotlib.pyplot as plt
from PIL import Image

import siamConfig

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image Transform
transform = transforms.Compose([transforms.Resize((siamConfig.height, siamConfig.width)), transforms.ToTensor()])

# Converts an image path to the corresponding image tensor
def imagePathToImage(imgPath=None):
    path = str(imgPath)

    img = Image.open(path)
    img = img.convert("L")

    if transform is not None:
        img = transform(img)
    return img

# Shows image Pairs
def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(
            75,
            8,
            text,
            style="italic",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 10},
        )
    plt.imshow(numpy.transpose(npimg, (1, 2, 0)))
    plt.show()