# Stores commonly-used variables needed for the model to function properly

# Imports

import torch
import torchvision.transforms as transforms # Needed for image transform

### Datasets ###

# Training
train_dir = "sets\\tkPieceImages\\" # Root Directory
train_csv = "csvFiles\\demoTrain.csv" # CSV File Pointing to Images (Format: Image,Class)

# Validation
eval_dir = "sets\\tkPieceImages\\" # Root Directory
eval_csv = "csvFiles\\demoEval.csv" # CSV File Pointing to Images (Format: Image,Class)

# Testing
test_dir = "sets\\tkPieceImages\\" # Root Directory
test_csv = "csvFiles\\demoTest.csv" # CSV File Pointing to Images (Format: Image,Class)

### Model Configuration Parameters ###

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image Transform Parameters
width = 105
height = 105
transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])

# Training Parameters
epochs = 20
loss_margin = 1.0

# Training Dataloader Parameters
train_shuffle = True
train_num_workers = 8
train_batch_size = 32

# Validation Dataloader Parameters

eval_shuffle = False
eval_num_workers = train_num_workers
eval_batch_size = train_batch_size

# Testing Parameters
max_tests = 10 # Any value < 1 causes a full test run

# Testing Dataloader Parameters
test_shuffle = False
test_num_workers = 6
test_batch_size = 1

# Model State Dictionary Path
model_path = "..\\savedModels\\demoModel.pth"
