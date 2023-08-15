# Stores commonly used variables for reusability

# Imports

# Used for defining the image transform
import torchvision.transforms as transforms
####################################################################################
# Configuration Variables

### Datasets ###

# Training
# Root Directory (MUST END IN "/"!!!)
training_dir = "/sets/demoImages/"
# Points to the images in directory that are being used. CSV Format: Relative Image Path, Image Class
training_csv = "/sets/demoTrainingCSV.csv"

# Validation
# Root Directory (MUST END IN "/"!!!)
validation_dir = "/sets/demoImages/"
# Points to the images in directory that are being used. CSV Format: Relative Image Path, Image Class
validation_csv = "/sets/demoValidationCSV.csv"

# Testing 
# Root Directory (MUST END IN "/"!!!)
testing_dir = "/sets/demoImages/"
# Points to the images in directory that are being used. CSV Format: Relative Image Path, Image Class
testing_csv = "/sets/demoTestingCSV.csv"

### Model Parameters ###

# Training and Validation Parameters
batch_size = 32 # Batch size for training and validation
epochs = 20 # Number of epochs to train over

# Size to resize images in Transform
width = 105
height = 105
# Image Transform (Used to convert imagaes to a format useable by the network)
transform = transform=transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])

# Contrastive Loss Margin
loss_margin = 1.0

# Max Number of Tests (Is overridden if greater than the length of the test CSV, as determined by pandas)
max_tests = 20

# Path used to save and load the model's state
model_path = "/savedModels/modelDemo.pth"

# File Types Accepted by Input Dialog
accepted_input_types = [("PNG files", "*.png"), ("JPG files", "*.jpg"), ("GIF files", "*.gif")]