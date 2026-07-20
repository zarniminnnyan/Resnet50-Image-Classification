import torch
import warnings


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

ROOT = "./root"
WEIGHT_PATH = "./models"

# Flags
INFERENCE = False
LR=1e-4  
NUM_EPOCHS=20

# Hyperparameter search ranges for Optuna
CONFIGS = {
    "lr_configs": {"low": 1e-5, "high": 1e-1},
    "weight_configs": {"low": 1e-6, "high": 1e-2},
    "num_epochs_configs": {"low": 10, "high": 50},
    "batch_configs": {"low": 32, "high": 256}
}
