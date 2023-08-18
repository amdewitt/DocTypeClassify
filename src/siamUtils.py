# Stores commonly-used functions needed for the model to function properly

# Imports
from PIL import Image
import siamConfig

import numpy as np
import matplotlib.pyplot as plt

# Converts an image path to the corresponding image tensor
def imagePathToImage(imgPath=None, transform = siamConfig.transform):
    path = str(imgPath)

    img = Image.open(path)
    img = img.convert("L")

    if transform is not None:
        img = transform(img)
    return img

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
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()