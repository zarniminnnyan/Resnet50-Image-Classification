import torch  
import tqdm
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
from configs import DEVICE,ROOT,WEIGHT_PATH,BATCH_SIZE,SPLIT_SIZE,SEED,INFERENCE,LR,NUM_EPOCHS

class Resnet50Classifier(pl.LightningModule): 
  def __init__(self,lr:float,num_classes:int,num_epochs:int):
    super().__init__()
    self.save_hyperparameters()      
    self.num_classes=num_classes
    
    self.model=load_resnet(DEVICE,WEIGHT_PATH,freeze_backbones=True)
    
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
    self.log("train_acc",train_acc,on_step=False,on_epoch=True)
    self.log("train_loss",loss,prog_bar=True)
    return loss
    
  def validation_step(self,batch,batch_idx):
    inputs,labels=batch 
    outputs=self(inputs)
    val_loss=F.cross_entropy(outputs,labels)
    val_acc=self.val_acc(outputs,labels)
    self.log("val_acc",val_acc, on_step=False,on_epoch=True,prog_bar=True)
    self.log("val_loss",val_loss,prog_bar=True)
    
  def configure_optimizers(self):
    optimizer=torch.optim.SGD(
      self.parameters(),
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

  if not INFERENCE:
      train_loader,val_loader=load_tv_dataset(cifar_100_path=ROOT,batch_size=BATCH_SIZE,finetuned_model_inference=INFERENCE,split_size=SPLIT_SIZE,seed=SEED)
  
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

  
  
if __name__=="__main__":
  main()





