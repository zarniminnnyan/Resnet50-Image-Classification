import torch
import json
import os

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import load_resnet
from data import load_tv_dataset
from utils import get_model_size, measure_inference_time
from configs import DEVICE, ROOT, WEIGHT_PATH, INFERENCE, LR, NUM_EPOCHS, BATCH_SIZE, SPLIT_SIZE, SEED,CONFIGS

def resnet_finetune_setup():
  # Load ResNet-50 and make only layer4 + fc trainable.
  # All other layers stay frozen to keep pretrained features fixed.
  trainable_layers=["fc","layer4"]
  model = load_resnet(DEVICE, WEIGHT_PATH)
  for name,param in model.named_parameters():
    if any(layer in name for layer in trainable_layers):
      param.requires_grad=True  
    else:
      param.requires_grad=False 
      
  return model 



class Resnet50Classifier(pl.LightningModule):
    """Lightning module for fine-tuning ResNet50 on CIFAR-100."""
    
    def __init__(self, learning_rate: float, num_classes: int, num_epochs: int, weight_decay: float = 5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.model=resnet_finetune_setup()
        self.trainable_params=[p for p in self.model.parameters() if p.requires_grad]
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        train_acc = self.train_acc(outputs, labels)

        self.log("train_acc", train_acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = F.cross_entropy(outputs, labels)
        val_acc = self.val_acc(outputs, labels)

        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.trainable_params,
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.num_epochs,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


def get_trainer(max_epochs, callbacks=None):
    """Helper to create trainer with consistent config."""
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks or []
    )


def get_callbacks(study=None):
    """Helper to create callbacks."""
    callbacks = [
        EarlyStopping(monitor='val_acc', patience=5, mode='max'),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            filename="finetuned-resnet50"
        )
    ]
    if study:
        callbacks.append(PyTorchLightningPruningCallback(study, monitor='val_acc'))
    return callbacks


def objective(trial):
    """Optuna objective function."""
    learning_rate = trial.suggest_float(
        'learning_rate',
        CONFIGS["lr_configs"]["low"],
        CONFIGS["lr_configs"]["high"],
        log=True
    )
    
    weight_decay = trial.suggest_float(
        'weight_decay',
        CONFIGS["weight_configs"]["low"],
        CONFIGS["weight_configs"]["high"],
        log=True
    )
    num_epochs = trial.suggest_int(
        'num_epochs',
        CONFIGS["num_epochs_configs"]["low"],
        CONFIGS["num_epochs_configs"]["high"]  
    )
    batch_size = trial.suggest_int(
        'batch_size',
        CONFIGS["batch_configs"]["low"],
        CONFIGS["batch_configs"]["high"]
    )

    model = Resnet50Classifier(
        learning_rate=learning_rate,
        num_classes=100, 
        num_epochs=num_epochs,
        weight_decay=weight_decay
    )
    
    train_loader, val_loader = load_tv_dataset(
        cifar_100_path=ROOT,
        batch_size=batch_size,
        finetuned_model_inference=INFERENCE,
        split_size=SPLIT_SIZE,
        seed=SEED
    )

    trainer = get_trainer(num_epochs, get_callbacks(trial.study))
    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics["val_acc"].item()


def run_optimization(n_trials=20):
    """Run hyperparameter optimization."""
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    return study


def resnet50_with_best_params():
    """Train model with best hyperparameters found by Optuna."""
    study = run_optimization()
    best_params = study.best_trial.params

    model = Resnet50Classifier(
        learning_rate=best_params["learning_rate"],
        num_classes=100,
        num_epochs=best_params["num_epochs"],
        weight_decay=best_params["weight_decay"]
    )

    train_loader, val_loader = load_tv_dataset(
        cifar_100_path=ROOT,
        batch_size=best_params["batch_size"],
        finetuned_model_inference=INFERENCE,
        split_size=SPLIT_SIZE,
        seed=SEED
    )

    trainer = get_trainer(best_params["num_epochs"])
    trainer.fit(model, train_loader, val_loader)
    
    trained_model = model.model
    results = {
        "Best Trial Value": study.best_trial.value,
        "Best Params": best_params,
        "Model Size (MB)": round(get_model_size(trained_model), 2),
        "Inference Time (ms)": round(measure_inference_time(trained_model, DEVICE), 2),
        "Accuracy (%)": round(trainer.callback_metrics["val_acc"].item(), 2)
    }

    save_results(results, filename="optuna_finetuned_results.json")
    print(f"Optuna results saved: {results}")
    return results


def save_results(results, filename="finetuned_results.json"):
    """Save results to JSON file."""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, filename)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)


def main(is_optuna: bool = False):
    """Main training function."""
    if is_optuna:
        resnet50_with_best_params()
        return

    if INFERENCE:
        raise RuntimeError("Inference mode not allowed. Set INFERENCE=False.")

    model = Resnet50Classifier(
        learning_rate=LR,
        num_classes=100,
        num_epochs=NUM_EPOCHS,
        weight_decay=5e-4  
    )
    train_loader, val_loader = load_tv_dataset(
        cifar_100_path=ROOT,
        batch_size=BATCH_SIZE,
        finetuned_model_inference=INFERENCE,
        split_size=SPLIT_SIZE,
        seed=SEED
    )

    trainer = get_trainer(NUM_EPOCHS, get_callbacks())
    trainer.fit(model, train_loader, val_loader)

    # Evaluate model
    trained_model = model.model
    results = {
        "Model Size (MB)": round(get_model_size(trained_model), 2),
        "Inference Time (ms)": round(measure_inference_time(trained_model, DEVICE), 2),
        "Accuracy (%)": round(trainer.callback_metrics["val_acc"].item(), 2)
    }

    save_results(results)
    print(f"Results saved: {results}")


if __name__ == "__main__":
    main()