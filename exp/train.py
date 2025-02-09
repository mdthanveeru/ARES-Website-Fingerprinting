import os
import sys
import torch
import random
import argparse
import numpy as np
from WFlib import models
from WFlib.tools import data_processor, model_utils

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)  #sets the seed for Python's built-in random module.
torch.manual_seed(fix_seed)  #sets the seed for PyTorch's random number generator.
np.random.seed(fix_seed)   #sets the seed for NumPy's random number generator.

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, required=True, default="CW", help="Dataset name")
parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")
parser.add_argument("--num_tabs", type=int, default=1, 
                    help="Maximum number of tabs opened by users while browsing")

# Input parameters
parser.add_argument("--train_file", type=str, default="train", help="Train file")
parser.add_argument("--valid_file", type=str, default="valid", help="Valid file")
parser.add_argument("--feature", type=str, default="DIR", 
                    help="Feature type, options=[DIR, DT, DT2, TAM, TAF]")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")

# Optimization parameters
parser.add_argument("--num_workers", type=int, default=10, help="Data loader num workers")
parser.add_argument("--train_epochs", type=int, default=30, help="Train epochs")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")
parser.add_argument("--learning_rate", type=float, default=2e-3, help="Optimizer learning rate")
parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
parser.add_argument("--loss", type=str, default="CrossEntropyLoss", help="Loss function")
parser.add_argument("--lradj", type=str, default="None", 
                    help="adjust learning rate, option=[None, StepLR]")

# Output parameters
parser.add_argument('--eval_metrics', nargs='+', required=True, type=str, 
                    help="Evaluation metrics, options=[Accuracy, Precision, Recall, F1-score, P@min, r-Precision]")
parser.add_argument("--save_metric", type=str, default="F1-score", 
                    help="Save the model when the metric reaches its maximum value on the validation set")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Location of model checkpoints")
parser.add_argument("--load_file", type=str, default=None, help="The pre-trained model file")
parser.add_argument("--save_name", type=str, default="base", help="Name used to save the model")

# Parse arguments
args = parser.parse_args()

# Ensure the specified device is available
if args.device.startswith("cuda"):
    assert torch.cuda.is_available(), f"The specified device {args.device} does not exist"
device = torch.device(args.device)

# Define paths for dataset and checkpoints
in_path = os.path.join("./datasets", args.dataset)
if not os.path.exists(in_path):
    raise FileNotFoundError(f"The dataset path does not exist: {in_path}")
ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
os.makedirs(ckp_path, exist_ok=True)

out_file = os.path.join(ckp_path, f"{args.save_name}.pth")
if os.path.exists(out_file):
    print(f"{out_file} has been generated.")
    sys.exit(1)

# Load training and validation data
print(f"loading train file: ", os.path.join(in_path, f"{args.train_file}.npz"))
train_X, train_y = data_processor.load_data(os.path.join(in_path, f"{args.train_file}.npz"), args.feature, args.seq_len, args.num_tabs)
valid_X, valid_y = data_processor.load_data(os.path.join(in_path, f"{args.valid_file}.npz"), args.feature, args.seq_len, args.num_tabs)

if args.num_tabs == 1:
    num_classes = len(np.unique(train_y))
    assert num_classes == train_y.max() + 1, "Labels are not continuous" # Ensure labels are continuous
else:
    num_classes = train_y.shape[1]

# Print dataset information
print(f"Train: X={train_X.shape}, y={train_y.shape}")
print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
print(f"num_classes: {num_classes}")

# Load data into iterators
train_iter = data_processor.load_iter(train_X, train_y, args.batch_size, True, args.num_workers)
valid_iter = data_processor.load_iter(valid_X, valid_y, args.batch_size, False, args.num_workers)

#Initialize model, optimizer, and loss function
model = eval(f"models.{args.model}")(num_classes) 
#model = models.ARES(100)
optimizer = eval(f"torch.optim.{args.optimizer}")(model.parameters(), lr=args.learning_rate)  
#optimzer = torch.optim.AdamW(model.parameters(), lr=0.001)
#initialize AdamW optimizer with learning rate 0.001

if args.load_file is None:
    print("No pre-trained model")
else:
    print("Loading the pretrained model in ", args.load_file)
    checkpoint = torch.load(args.load_file)

    for k in list(checkpoint.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                checkpoint[k[len("backbone."):]] = checkpoint[k]
        del checkpoint[k]

    log = model.load_state_dict(checkpoint, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

model.to(device) # Move model to GPU

# Train the model
model_utils.model_train(
    model, #our model is ARES
    optimizer, #for updating model weights(AdamW)
    train_iter, #Training dataset (in batches).
    valid_iter, #Validation dataset (in batches).
    args.loss, #loss function(MultiLabelSoftMarginLoss)
    args.save_metric, #The metric used to decide the best model (e.g., AP@2).
    args.eval_metrics, #List of evaluation metrics (e.g., ["AUC", "P@2", "AP@2"]).
    args.train_epochs, #Number of epochs to train.
    out_file, #File path to save the best model.
    num_classes, #Number of classes for classification.
    args.num_tabs, #No. of tabs
    device, #cpu or gpu(cuda)
    args.lradj #Learning rate adjustment strategy.
)