import torch  
import tqdm
from  model import  load_resnet
from data import load_tv_dataset
from utils import get_model_size,measure_inference_time
from configs import DEVICE,ROOT,BATCH_SIZE,SPLIT_SIZE,SEED,INFERENCE

@torch.no_grad()
def evaluate_baseline_model(model,eval_loader,device):
    correct = 0
    total = 0
    model.eval()
    tqdm_eval_loader=tqdm(eval_loader,leave=False,desc="Evaluating the pretrained Resnet50 baseline")
    
    for img,label  in tqdm_eval_loader: 
        img,label=img.to(device),label.to(device)
        outputs=model(img)
        _,predicted=torch.max(outputs,dim=1)
        
        total+=label.size(0)
        correct+=(predicted==label).sum().item()
    accuracy=100*correct/total 
    return accuracy 

def main():
    if INFERENCE:
        inference_loader=load_tv_dataset(root=ROOT,batch_size=BATCH_SIZE,finetuned_model_inference=INFERENCE,split_size=SPLIT_SIZE,seed=SEED)
    else:
        train_loader,val_loader=load_tv_dataset(root=ROOT,batch_size=BATCH_SIZE,finetuned_model_inference=INFERENCE,split_size=SPLIT_SIZE,seed=SEED)
        
    resnet50=load_resnet(DEVICE)
    model_size=get_model_size(resnet50)
    inference_time=measure_inference_time(resnet50,DEVICE)
    model_accuracy=evaluate_baseline_model(resnet50,val_loader,DEVICE)
    
    print(f"\nResNet50 Baseline Results:")
    print("-"*35)
    print(f"  • Model Size       : {model_size}")
    print(f"  • Inference Time   : {inference_time}")
    print(f"  • Accuracy         : {model_accuracy:.2f}%")

if __name__=="__main__":
    main()
      

        
        
        
        
        
    
    