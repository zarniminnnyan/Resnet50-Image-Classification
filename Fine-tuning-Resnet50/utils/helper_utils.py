import matplotlib.pyplot as plt 
import tempfile 
import torch  
import time 
import os 

def get_model_size(model):
    """
    Create the temporary file path to store model with tempfile 
    Get model size in MB
    
    Args:
    model: Resnet50 Pretrained model
    
    Return:
    size_mb: convert the model size bytes to MB
    
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tempFile: 
        tempfile_path = tempFile.name 
        #Save model dict 
        torch.save(model.state_dict(), tempfile_path)
    
    #Get Resnet50 baseline size   
    model_size_bytes = os.path.getsize(tempfile_path)
    #Get model size in Bytes to MB
    size_mb = model_size_bytes / (1024 * 1024)
    #Remove temporary file path
    os.remove(tempfile_path)
    return size_mb

def measure_inference_time(model,device,input_shape=(1,3,32,32),warm_up_runs=10,num_runs=100):
    """
    Measure the inference time of baseline model and  Fine-tuned Resnet50
    
    Returns:
    avg_time: Average inference timing of the model
    """
    #Create sample input data
    sample_input=torch.rand(input_shape).to(device)
    model.eval()
    #Run warmup for initializing  system caches and operations
    with torch.no_grad():
        for _ in range(warm_up_runs):
            model(sample_input)
    
    timings=[]
    with torch.no_grad():
        for _ in range(num_runs):
            #Record start time 
            time_start=time.time()
            #feed sample data to the model
            model(sample_input)
            time_end=time.time()
            timings.append(time_end-time_start)
        #Get the total size and total of the timings
        timings_size=len(timings)
        total_timings=sum(timings)
        #Calculate avg time with average formula
        avg_time=total_timings/timings_size
        return avg_time 
    

            
    
    
    
