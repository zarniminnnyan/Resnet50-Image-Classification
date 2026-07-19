import configs
import torch
import warnings
import sys

# Detect if running in Colab
IS_COLAB = "google.colab" in sys.modules

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Warn if CUDA is not available
if not torch.cuda.is_available():
    warnings.warn(
        "CUDA is not available. Running on CPU will be significantly slower.",
        UserWarning
    )

# Dataset settings
BATCH_SIZE = 32
SPLIT_SIZE = 0.8
SEED = 42

# Model weights and dataset paths
if IS_COLAB:
    ROOT = "/content/drive/MyDrive/cifar100-datasets"
    WEIGHT_PATH = "/content/drive/MyDrive/models"
else:
    ROOT = "/content/drive/MyDrive/cifar100-datasets"
    WEIGHT_PATH = "./models"

# Flags
INFERENCE = False
LR=1e-3     
NUM_EPOCHS=20

