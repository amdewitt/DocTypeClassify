from tkinter import filedialog
#from PIL import Image
import pandas

from siamDataset import PairwiseDataset
from siamModel import SiameseModel
import siamConfig
import siamUtils

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

net = SiameseModel().to(siamUtils.device)

def __main__():
    print("Enter Root Directory")
    rootDir = filedialog.askdirectory(mustexist=True)
    print("Open Image Set CSV (Must have at least 2 images)")
    paths_csv = filedialog.askopenfilename(title="Open CSV (No header, Format: relativeImagePath)", filetypes=[("CSV files", "*.csv")])

    # Variables

    dataset = PairwiseDataset(
        csvFile = paths_csv,
        directory = rootDir,
        transform = siamUtils.transform,
    )

    dataloader = DataLoader(
        dataset,
        shuffle = siamConfig.test_shuffle,
        num_workers = siamConfig.test_num_workers,
        batch_size = siamConfig.test_batch_size
    )

    print("--------------------")
    if(dataset.__len__() < 2):
        "ERROR: CSV must have at least 2 rows."
        return
    for i, data in enumerate(dataloader, 0):
        img0, img1 = data
        concat = torch.cat((img0,img1),0)
        img0, img1 = img0.to(siamUtils.device), img1.to(siamUtils.device)

        output0, output1 = net(img0, img1)
        euclidean_distance = F.pairwise_distance(output0, output1)

        img0_path, img1_path = dataset.__getItemPaths__(i)

        output0, output1 = net(img0, img1)
        euclidean_distance = F.pairwise_distance(output0, output1)        

        if euclidean_distance.item() > siamConfig.dissimilar_prediction_threshold and siamConfig.include_threshold == False:
            predLabel = "Dissimilar"
        elif euclidean_distance.item() >= siamConfig.dissimilar_prediction_threshold:
            predLabel = "Dissimilar"
        else:
            predLabel = "Similar"
        print(f"Image 0: {img0_path}\nImage 1: {img1_path}\nPairwise Distance: {euclidean_distance.item()}\nLabel: {predLabel}\n--------------------")
        siamUtils.imshow(torchvision.utils.make_grid(concat))

if __name__ == "__main__":
    __main__()
