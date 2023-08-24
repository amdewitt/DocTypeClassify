# Stores commonly-used variables needed for the model to function properly

### Datasets ###

# Training
train_dir = "sets\\mnemPieceImages\\" # Root Directory
train_csv = "csvFiles\\demoTrain.csv" # CSV File Pointing to Images (Format: Image,Class)

# Validation
eval_dir = "sets\\mnemPieceImages\\" # Root Directory
eval_csv = "csvFiles\\demoEval.csv" # CSV File Pointing to Images (Format: Image,Class)

# Testing
test_dir = "sets\\mnemPieceImages\\" # Root Directory
test_csv = "csvFiles\\demoTest.csv" # CSV File Pointing to Images (Format: Image,Class)

# Classification
classify_dir = "sets\\mnemPieceImages\\" # Root Directory
classify_csv = "csvFiles\\demoClassify.csv" # CSV File Pointing to Images (Format: Image,Class)

### Model Configuration Parameters ###

# CSV Reader Parameters
treat_first_line_as_header = False

# Image Transform Parameters
width = 105
height = 105

# Training Parameters
epochs = 20
loss_margin = 1.0
save_on_new_best_loss = True

# Training Dataloader Parameters
train_shuffle = True
train_num_workers = 8
train_batch_size = 32

# Validation Dataloader Parameters

eval_shuffle = False
eval_num_workers = train_num_workers
eval_batch_size = train_batch_size

# Testing Parameters
max_tests = 0 # Any value < 1 causes a full test run
dissimilar_prediction_threshold = 0.5
include_threshold = False

# Testing Dataloader Parameters
test_shuffle = False
test_num_workers = 0
test_batch_size = 1

# Model State Dictionary Path
model_path = "..\\savedModels\\demoModel.pth"

# Accepted Input Types
accepted_input_types = [("PNG files", "*.png"),("JPG files", "*.jpg"),("GIF files", "*.gif")]