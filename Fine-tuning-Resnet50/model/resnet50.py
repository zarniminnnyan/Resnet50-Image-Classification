import torch 
import torchvision.models  as tv 
import os 



def  load_resnet(device:torch.device | str,model_path:str ):
    """
    Load a pretrained ResNet-50 model.

    Args:
        device (torch.device | str): Target device ("cuda" or "cpu").

    Returns:
        nn.Module: Pretrained ResNet-50 on the specified device.
    """
    if os.path.exists(model_path):
        model=tv.resnet50(weights=None) 
        model.load_state_dict(torch.load(model_path,map_location=device))
 
    else:
        model=tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT) 
        torch.save(model.state_dict(),model_path)   
    model.to(device)
    return model 

