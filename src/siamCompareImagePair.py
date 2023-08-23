import tkinter
from tkinter import ttk
from tkinter import filedialog
from PIL import Image

from siamModel import SiameseModel
import siamConfig

import torch
import torch.nn.functional as F
import torchvision

import siamUtils

net = SiameseModel().to(siamUtils.device)

def __main__():
    print("Open Image 0")

    root = filedialog.Tk()
    root.withdraw()
    path0 = filedialog.askopenfilename(mode='r', filetypes=siamConfig.accepted_input_types)
    root.destroy()

    print("Open Image 1")
    root = filedialog.Tk()
    root.withdraw()
    path1 = filedialog.askopenfilename(mode='r', filetypes=siamConfig.accepted_input_types)
    root.destroy()

    print(f"Image 0: {path0}\nImage 1: {path1}")
    img0 = siamUtils.imagePathToImage(path0)
    img1 = siamUtils.imagePathToImage(path1)
    concat = torch.cat((img0,img1),0)
    img0, img1 = img0.to(siamUtils.device), img1.to(siamUtils.device)

    output0, output1 = net(img0, img1)
    euclidean_distance = F.pairwise_distance(output0, output1)

    print(f"Euclideam Distance: {euclidean_distance.item()}")

    if euclidean_distance.item() > siamConfig.dissimilar_prediction_threshold and siamConfig.include_threshold == False:
        predLabel = "Dissimilar"
    elif euclidean_distance.item() >= siamConfig.dissimilar_prediction_threshold:
        predLabel = "Dissimilar"
    else:
        predLabel = "Similar"
    print(f"Label: {predLabel}")
    siamUtils.imshow(torchvision.utils.make_grid(concat))

if __name__ == "__main__":
    __main__()
