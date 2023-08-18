# Stores commonly-used functions needed for the model to function properly

# Imports
from PIL import Image
import siamConfig

# Converts an image path to the corresponding image tensor
def imagePathToImage(imgPath=None, transform = siamConfig.transform):
    path = str(imgPath)

    img = Image.open(path)
    img = img.convert("L")

    if transform is not None:
        img = transform(img)
    return img