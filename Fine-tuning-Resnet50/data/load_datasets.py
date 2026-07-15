from torch.utils.data import DataLoader,random_split
from torchvision.datasets import ImageFolder,CIFAR100
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import os 


def build_transforms():
    """
    Build training augmentation and validation/inference transforms.
    """

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform

    
class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.indices)


def load_tv_dataset(root: str, batch_size: int, finetuned_model_inference: bool, split_size: float, seed: int):
    """ 
    Load a torchvision dataset or inference images. 
    
    Args: dataset (str): Name of the torchvision "cifar100" dataset.
    batch_size (int): Number of samples per batch. 
    finetuned_model_inference (bool): If True, load images from inference_data using ImageFolder; otherwise, load the specified torchvision dataset.
    Returns:
   DataLoader: A single DataLoader for inference or training and validation DataLoaders for dataset mode.
   
    """
    # Load the transforms 
    train_transform, val_transform = build_transforms()
    
    # Load the images from inference_data if inference is true
    if finetuned_model_inference:
        inference_datasets = ImageFolder(
            root="inference_data/",
            transform=val_transform
        )
        
        inference_loader = DataLoader(
            inference_datasets,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
        return inference_loader
   
    else: 
        cifar_100_path = os.path.join(root, "cifar-100-dataset")
        
        if os.path.exists(cifar_100_path):
            download_flag=False  
        else:
            download_flag=True
    
        full_dataset = CIFAR100(
            root=cifar_100_path,
            train=True,
            download=download_flag, 
            transform=None 
        )
        
        # Calculate split sizes
        train_size = int(split_size * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_indices, val_indices = random_split(
            range(len(full_dataset)),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        #  Use the wrapper to cleanly isolate the augmentation pipelines
        train_dataset = TransformedSubset(full_dataset, train_indices, transform=train_transform)
        val_dataset = TransformedSubset(full_dataset, val_indices, transform=val_transform)
        
        # Create dataloaders for val and train
        train_loader = DataLoader( 
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True  
        )
        
        val_loader = DataLoader( 
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True 
        ) 
        
        return train_loader, val_loader
    
   
    
    
    