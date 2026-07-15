import torch 
import torchvision.models  as tv 

def  load_resnet(device:torch.device | str ):
    """
    Load a pretrained ResNet-50 model.

    Args:
        device (torch.device | str): Target device ("cuda" or "cpu").

    Returns:
        nn.Module: Pretrained ResNet-50 on the specified device.
    """
    model=tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT) 
    model=model.to(device)
    return model 

