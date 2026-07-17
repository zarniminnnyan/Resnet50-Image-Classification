import torch  
import tqdm
import json
import os 
from  model import  load_resnet
from data import load_tv_dataset
from utils import get_model_size,measure_inference_time
from configs import DEVICE,ROOT,WEIGHT_PATH,BATCH_SIZE,SPLIT_SIZE,SEED,INFERENCE

def customize_resnet50_for_finetuning(freeze_backbones:bool,num_classes:int):
  """
  Prepare Resnet50 for finetuning
  Args:

  freeze_backbon: If True, freeze backbones.
  num_classes: Num classes to predict

  Returns:
  model(nn.Module):Ready for finetuning

  """
  model=load_resnet(DEVICE,WEIGHT_PATH)
  if freeze_backbones:
    for name,param in model.named_parameters():
      param.requires_grad=False
    #Replace with the num classes of CIFAR100
    num_features=model.fc.in_features
    model.fc=torch.nn.Linear(num_features,num_classes)

  return model 

def main():
    model=customize_resnet50_for_finetuning(freeze_backbones=True,num_classes=100)
    print(model)
if __name__=="__main__":
  main()



