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
BATCH_SIZE = 64
SPLIT_SIZE = 0.8
SEED = 42

# Kaggle paths
ROOT = "/kaggle/input/cifar-100-python"       
WEIGHT_PATH = "/kaggle/working/models"       
LIGHTNING_LOGS_PATH = "/kaggle/working/logs
MODEL_CHECKPOINT_PATH = "/kaggle/working/checkpoints"

# Flags
INFERENCE = False
LR=1e-4
NUM_EPOCHS=20 

OPTUNA_EPOCHS = 10

# Hyperparameter search ranges for Optuna
CONFIGS = {
    "lr_configs": {"low": 1e-4, "high": 1e-3},
    "weight_configs": {"low": 1e-5, "high": 1e-3}
}
