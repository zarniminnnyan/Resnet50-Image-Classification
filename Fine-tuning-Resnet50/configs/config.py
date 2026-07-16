import torch
import warnings 

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set warning if CUDA is not available 
if not torch.cuda.is_available():
    warnings.warn("CUDA is not available. Running on CPU will be significantly slower.", UserWarning)
    
# Dataset settings
ROOT = "./root"
BATCH_SIZE = 32
SPLIT_SIZE = 0.8
SEED = 42

# Flags
INFERENCE = False
