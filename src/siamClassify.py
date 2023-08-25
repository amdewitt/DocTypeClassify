
# Imports

from tkinter import filedialog
#from PIL import Image
#import pandas

from siamDataset import PointwiseDataset
from siamModel import SiameseModel
import siamConfig
import siamUtils

import torch
import torch.nn.functional as F
#import torchvision
from torch.utils.data import DataLoader

net = SiameseModel().to(siamUtils.device)

def __main__():
    print("Open Image Set")
    paths = filedialog.askopenfilenames(filetypes=siamConfig.accepted_input_types)

    net.load_state_dict(torch.load(siamConfig.model_path))
    print("-"*20)
    for i, path in enumerate(paths, 0):
        dataset = PointwiseDataset(
            inputPath = path,
            csvFile = siamConfig.test_csv,
            directory = siamConfig.test_dir,
            transform = siamUtils.transform,
        )

        dataloader = DataLoader(
            dataset,
            shuffle = siamConfig.test_shuffle,
            num_workers = siamConfig.test_num_workers,
            batch_size = siamConfig.test_batch_size
        )

        best_class = ""
        best_pairwise_distance = 999999999
        for j, data in enumerate(dataloader, 0):
            inputImg, img0, class0 = data
            inputImg, img0 = inputImg.to(siamUtils.device), img0.to(siamUtils.device)
            output0, output1 = net(inputImg, img0)
            euclidean_distance = F.pairwise_distance(output0, output1)
            if euclidean_distance < best_pairwise_distance:
                best_pairwise_distance = euclidean_distance
                best_class = class0
        print(f"{i} : {path}")
        print(f"Class: {best_class[0]}")
        print("-"*20)


if __name__ == "__main__":
    __main__()