import torch  
import json
import os 
import optuna  
import torch.nn.functional as F      
import pytorch_lightning as pl     
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import Accuracy 
from  model import  load_resnet
from data import load_tv_dataset
from utils import get_model_size,measure_inference_time
from torch.optim.lr_scheduler import CosineAnnealingLR
from configs import DEVICE,ROOT,WEIGHT_PATH,INFERENCE,LR,NUM_EPOCHS,BATCH_SIZE,SPLIT_SIZE,SEED

class Resnet50Classifier(pl.LightningModule): 
  """
Lightning module for fine-tuning ResNet50 on CIFAR-100.

Handles:
- forward pass
- training step
- validation step
- optimizer configuration
"""
  def __init__(self,lr:float,num_classes:int,num_epochs:int):
    super().__init__()
    self.save_hyperparameters()      
    self.num_classes=num_classes
    
    self.model=load_resnet(DEVICE,WEIGHT_PATH)
    
    self.train_acc=Accuracy(task='multiclass',num_classes=self.num_classes)
    self.val_acc=Accuracy(task='multiclass',num_classes=self.num_classes)
  
  def forward(self,x):
    return self.model(x)
  
  def training_step(self,batch,batch_idx):
    inputs,labels=batch 
    outputs=self(inputs)
    loss=F.cross_entropy(outputs,labels) 
    #Compute the accuracy 
    train_acc=self.train_acc(outputs,labels)
    
    self.log(
      "train_acc",
      train_acc,
      on_step=False,
      on_epoch=True
      )
    self.log(
      "train_loss",
      loss,
      on_step=False,
      on_epoch=True,
      prog_bar=True

      )
    return loss
    
  def validation_step(self,batch,batch_idx):
    inputs,labels=batch 
    outputs=self(inputs)
    val_loss=F.cross_entropy(outputs,labels)
    val_acc=self.val_acc(outputs,labels)
    self.log(
      "val_acc",
      val_acc,
      on_step=False,
      on_epoch=True,
      prog_bar=True
      )
    self.log(
      "val_loss",
       val_loss,
       on_step=False,
        on_epoch=True,
        prog_bar=True
        )
        
  def configure_optimizers(self):
    optimizer=torch.optim.SGD(
      self.model.fc.parameters(),
      lr=self.hparams.lr,
      momentum=0.9,
       weight_decay=5e-4
      )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=self.hparams.num_epochs,     
        eta_min=1e-6
    )
    return {"optimizer": optimizer,
        "lr_scheduler": {
          "scheduler":scheduler,
          "interval":"epoch"
          
        }
        }
  

def main():
  
  model=Resnet50Classifier(lr=LR,num_classes=100,num_epochs=NUM_EPOCHS)

  if INFERENCE:
    raise RuntimeError("Inference mode is not allowed for this file. Set INFERENCE=False to proceed.")
  
  train_loader,val_loader=load_tv_dataset(
    cifar_100_path=ROOT,
    batch_size=BATCH_SIZE,
    finetuned_model_inference=INFERENCE,
    split_size=SPLIT_SIZE,
    seed=SEED
    )
  
  checkpoint = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=1,
    filename="finetuned-resnet50"
)
  trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    accelerator="auto",
    devices=1,
    callbacks=[checkpoint]
)
  trainer.fit(model,train_loader,val_loader)
     
  #trained model 
  trained_model=model.model  
  
  model_size=get_model_size(trained_model)
  inference_time=measure_inference_time(trained_model,DEVICE)
  accuracy=trainer.callback_metrics["val_acc"]
  
  finetuned_results = {
    "Model Size (MB)": round(model_size,2),
    "Inference Time (ms)": round(inference_time,2),
    "Accuracy (%)": round(accuracy.item(),2)
    }
    
    # Get the absolute path to the results directory
  results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
  os.makedirs(results_dir, exist_ok=True)

    # Full path to the JSON file
  json_path = os.path.join(results_dir, "finetuned_results.json")
    #Write the results to json file
  with open(json_path,"w") as f:
    json.dump(finetuned_results,f,indent=4)

if __name__=="__main__":
  main()





