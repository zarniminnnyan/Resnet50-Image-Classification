import torch

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset settings
ROOT = "./root"
BATCH_SIZE = 32
SPLIT_SIZE = 0.8
SEED = 42

# Flags
INFERENCE = False
