# Imports

import siamConfig
import siamUtils
from siamDataset import SiameseDataset
from torch.utils.data import DataLoader
from siamModel import SiameseModel

import torch
import torch.nn.functional as F
import torchvision

# Variables

test_dataset = SiameseDataset(
    csvFile = siamConfig.test_csv,
    directory = siamConfig.test_dir,
    transform = siamConfig.transform,
)

test_dataloader = DataLoader(
    test_dataset,
    shuffle=siamConfig.test_shuffle,
    num_workers = siamConfig.test_num_workers,
    batch_size = siamConfig.test_batch_size
)

net = SiameseModel().to(siamConfig.device)

def __main__():
    net.load_state_dict(torch.load(siamConfig.model_path))
    count = 0
    for i, data in enumerate(test_dataloader, 0):
        img0, img1, class0, class1 = data
        label = 1
        if class0 == class1:
            label = 0
        img0, img1 = img0.to(siamConfig.device), img1.to(siamConfig.device)

        output0, output1 = net(img0, img1)
        euclidean_distance = F.pairwise_distance(output0, output1)

        printLabel = "Dissimilar"
        if label == 0:
            printLabel = "Similar"

        siamUtils.imshow(torchvision.utils.make_grid(concat))
        print(f"Predicted Euclideam Distance: {euclidean_distance.item()}")
        print(f"Actual Label: {printLabel}")
        count += 1
        if count >= siamConfig.max_tests:
            break

if __name__ == "__main__":
    __main__()