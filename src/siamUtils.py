# Stores commonly-used functions needed for the model to function properly

# Imports
import os
from PIL import Image
import siamConfig

# Converts an image path to the corresponding image tensor
def imagePathToImage(self, imgPath=None, transform = siamConfig.transform, rootDir = None):
    if rootDir is not None:
        path = str(os.path.join(rootDir, imgPath))
    else:
        path = str(imgPath)

    img = Image.open(path)
    img = img.convert("L")

    if transform is not None:
        img = transform(img)
    return img