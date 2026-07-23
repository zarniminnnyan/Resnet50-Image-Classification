import os
import torch
from data import load_tv_dataset
from scripts.train_and_val_finetune import Resnet50Classifier,get_callbacks,get_trainer,save_results
from utils import get_model_size, measure_inference_time
from configs import (
    ROOT,
    INFERENCE,
    NUM_EPOCHS,
    BATCH_SIZE,
    SPLIT_SIZE,
    SEED,
    MODEL_CHECKPOINT_PATH, DEVICE,
    BEST_CHECKPOINT_PATH
)


def resume(path:str):
    
    
     #Load the cifar100 dataset
    train_loader, val_loader = load_tv_dataset(
        cifar_100_path=ROOT,
        batch_size=BATCH_SIZE,
        finetuned_model_inference=INFERENCE,
        split_size=SPLIT_SIZE,
        seed=SEED,
    )
    #Load model from the best checkpoint 
    model=Resnet50Classifier.load_from_checkpoint(path)
        
    trainer=get_trainer(NUM_EPOCHS,get_callbacks())
    trainer.fit(model,train_loader,val_loader,ckpt_path=path)
    best_model_path = trainer.checkpoint_callback.best_model_path
    
    #Safety check 
    if not best_model_path:
        best_model_path = path
    
    print(f"Loading best model from: {best_model_path}")
    best_model = Resnet50Classifier.load_from_checkpoint(
        best_model_path
    )
    best_model.to(DEVICE)
    best_model.eval()
    torch.save(
    best_model.model.state_dict(),
    os.path.join(
            MODEL_CHECKPOINT_PATH,
            "finetuned-resnet50-weights.pth",
        ),
    )
    results = {
        "Model Size (MB)": round(
            get_model_size(best_model.model),
            2,
        ),
        "Inference Time (ms)": round(
            measure_inference_time(best_model.model, DEVICE),
            2,
        ),
        "Validation Accuracy (%)": round(
            trainer.callback_metrics["val_acc"].item() * 100,
            2,
        ),
    }

    save_results(
        results,
        filename="resume_finetuned_results.json",
    )

    print(f"resume results saved: {results}")

    return results

def main():
    #Execute the resume training 
    resume_result=resume(BEST_CHECKPOINT_PATH)
    print("Resume training finished!")
    print("\n\n")
    print(resume_result)

if __name__=="__main__":
    main()
    
        
    