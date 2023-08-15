# Stores commonly used variables for reusability

# Imports

# Used for defining the image transform
import torchvision.transforms as transforms
####################################################################################
# Configuration Variables

# Training
training_csv = "/sets/train/" # Points to the images in directory that are being used. CSV Format: Image Path (Relative to Root Directory), Class
training_dir = "/sets/trainCSV.csv" # Root Directory (MUST END IN "/"!!!)

# Validation
validation_csv = "/sets/validation/" # Points to the images in directory that are being used. CSV Format: Image Path (Relative to Root Directory), Class
validation_dir = "/sets/validationCSV.csv" # Root Directory (MUST END IN "/"!!!)

# Testing 
testing_csv = "/sets/test/" # Points to the images in directory that are being used. CSV Format: Image Path (Relative to Root Directory), Class
testing_dir = "/sets/testCSV.csv" # Root Directory (MUST END IN "/"!!!)

# Training and Validation Parameters
batch_size = 32 # Batch size for training and validation
epochs = 20 # Number of epochs to train over

# Size to resize images in Transform
width = 400
height = 400
# Image Transform (Used to convert imagaes to a format useable by the network)
transform = transform=transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])

# Contrastive Loss Margin
loss_margin = 1.0

# Max Number of Tests (Is overridden if greater than the length of the test CSV, as determined by pandas)
max_tests = 20

# Path used to save and load the model's state
model_path = "/savedModels/model.pth"

# File Types Accepted by Input Dialog
accepted_input_types = [("PNG files", "*.png"), ("JPG files", "*.jpg"), ("GIF files", "*.gif")]