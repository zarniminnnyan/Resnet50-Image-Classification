import torch 
import torch.nn as nn
import torchvision.models  as tv 
import os 



def  load_resnet(device:torch.device | str,weight_dir:str,num_classes=100):
    """
    Load a pretrained ResNet-50 model.

    Args:
        device (torch.device | str): Target device ("cuda" or "cpu").

    Returns:
        nn.Module: Pretrained ResNet-50 on the specified device.
    """
    model_path=os.path.join(weight_dir,"baseline_weight.pth")
    
    os.makedirs(weight_dir, exist_ok=True)
    if os.path.exists(model_path):
        model=tv.resnet50(weights=None) 
        model.load_state_dict(torch.load(model_path,map_location=device),strict=False)
 
    else:
        model=tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT) 
        torch.save(model.state_dict(),model_path) 

    #Freeze params and replace the cifar 100 classes
    for name,param in model.named_parameters():
        param.requires_grad=False 
        
    in_features=model.fc.in_features
    model.fc=nn.Linear(in_features=in_features,out_features=num_classes)
    model.to(device)
    return model 

