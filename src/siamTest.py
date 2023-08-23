# Imports

import siamConfig
import siamUtils
from siamDataset import SiameseDataset
from torch.utils.data import DataLoader
from siamModel import SiameseModel

import torch
import torch.nn.functional as F
import torchvision

import numpy
import matplotlib.pyplot as plt

# Variables

test_dataset = SiameseDataset(
    csvFile = siamConfig.test_csv,
    directory = siamConfig.test_dir,
    transform = siamUtils.transform,
)

test_dataloader = DataLoader(
    test_dataset,
    shuffle=siamConfig.test_shuffle,
    num_workers = siamConfig.test_num_workers,
    batch_size = siamConfig.test_batch_size
)

net = SiameseModel().to(siamUtils.device)

def __main__():
    net.load_state_dict(torch.load(siamConfig.model_path))
    count = 0
    for i, data in enumerate(test_dataloader, 0):
        img0, img1, label, class0, class1 = data
        concat = torch.cat((img0,img1),0)
        img0, img1, label = img0.to(siamUtils.device), img1.to(siamUtils.device), label.to(siamUtils.device)

        output0, output1 = net(img0, img1)
        euclidean_distance = F.pairwise_distance(output0, output1)

        img0_path, img1_path = test_dataset.__getItemPaths__(i)
        print(f"Image 0: {img0_path}, Class of Image 0: {class0}\nImage 1: {img1_path}, Class of Image 1: {class1}")
        print(f"Predicted Euclideam Distance: {euclidean_distance.item()}")

        if euclidean_distance.item() > siamConfig.dissimilar_prediction_threshold and siamConfig.include_threshold == False:
            predLabel = "Dissimilar"
        elif euclidean_distance.item() >= siamConfig.dissimilar_prediction_threshold:
            predLabel = "Dissimilar"
        else:
            predLabel = "Similar"
        print(f"Predicted Label: {predLabel}")

        if label == torch.FloatTensor([[0]]).to(siamUtils.device):
            printLabel = "Similar"
        else:
            printLabel = "Dissimilar"
        print(f"Actual Label: {printLabel}")
        print("-"*20)
        siamUtils.imshow(torchvision.utils.make_grid(concat))
        if siamConfig.max_tests > 0:
            count += 1
            if count >= siamConfig.max_tests:
                break

if __name__ == "__main__":
    __main__()