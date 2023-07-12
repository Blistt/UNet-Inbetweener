from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_class import MyDataset
import unet_int
import tqdm
from collections import defaultdict
import torch
import torchmetrics
import my_metrics
import chamfer_dist
import numpy as np
from utils import visualize_batch_eval, write_log
from torchvision.utils import save_image
import os

'''
Main Evaluation Function
    Evaluates a dataset given a set of metrics, and a model with weights already loaded
'''
def evaluate(dataset, model, metrics, epoch=0, batch_size=8, display_step=10, device='cuda:1', experiment_dir='exp/', train_test='testing'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    results = defaultdict(list)
    for input1, labels, input2 in tqdm.tqdm(dataloader):
        # Flattens the image
        input1, labels, input2 = input1.to(device), labels.to(device), input2.to(device)

        with torch.no_grad():
            pred = model(input1, input2)
            pred = torch.sigmoid(pred)  # THIS IS SOMETHING TO PAY ATTENTION TO, IT MIGHT BE NEEDED EVERYWHERE ELSE OR REMOVED
            
            # Saves images of predictions and real images
            if epoch % display_step == 0:
                save_image((labels>0.5).float(), experiment_dir + 'eval_' + str(epoch) + '_true.png', nrow=2, normalize=False)
                save_image((pred>0.5).float(), experiment_dir + 'eval_' + str(epoch) + '_preds.png', nrow=2, normalize=False)
            
            raw_metrics = metrics(pred, labels) # Computes metrics
        for k,v in raw_metrics.items():
            results[k].append(v.item())

        # Store metrics log in a csv file
        write_log(results, experiment_dir, train_test)
    
    # Returns the average over all batches in the epoch for each metric
    return {k: np.mean(results[k]) for k,v in results.items()}


'''
Evaluates over multiple checkpoints
    Evaluates a model over the course of multiple checkpoints already stored as pth files
'''
def evaluate_multiple_checkpoints(dataset, model, metrics, checkpoints_dir, batch_size=8, device='cuda:1', experiment_dir='exp/', display_step=10, train_test='testing'):
    
    # Makes experiment dirs if they don't exist
    if not os.path.exists(experiment_dir + train_test): os.makedirs(experiment_dir + train_test)

    checkpoints = [file for file in os.listdir(checkpoints_dir) if file.endswith('.pth')]   # Filters for only .pth files
    results = defaultdict(list)

    # Iterates over all checkpoints in directory
    for i, checkpoint in enumerate(checkpoints):
        model.load_state_dict(torch.load(checkpoints_dir + checkpoint))
        model = model.eval()
        checkpoint_metrics = evaluate(dataset, model, metrics, i, batch_size=batch_size, device=device, experiment_dir=experiment_dir,
                                    display_step=display_step, train_test='testing')
        for k, v in checkpoint_metrics.items():
            results[k].append(v.item())
        
        # Plots metrics
        visualize_batch_eval(results, i, experiment_dir=experiment_dir, train_test=train_test)

        print('Epoch:', i, checkpoint_metrics)

            


# Main function
if __name__ == '__main__':
    torch.cuda.set_device(1)
    device = torch.device('cuda:1')

    '''
    Dataset parameters
    '''
    data_dir = '/data/farriaga/atd_12k/Line_Art/test_2k_original/'
    input_dim = 2
    label_dim = 1
    initial_shape = (512, 512)
    target_shape = (373, 373)
    binary_threshold = 0.75
    dataset = MyDataset(data_dir, transform=transforms.ToTensor(), resize_to=initial_shape, binarize_at=binary_threshold,
                         crop_shape=target_shape)
    batch_size = 32


    '''
    Model parameters
    '''
    model = unet_int.UNet(input_dim, label_dim).to(device)
    metrics = torchmetrics.MetricCollection({
        'psnr': my_metrics.PSNRMetricCPU(),
        'ssim': my_metrics.SSIMMetricCPU(),
        'chamfer': chamfer_dist.ChamferDistance2dMetric(binary=0.5),
        'mse': torchmetrics.MeanSquaredError(),
    }).to(device).eval()
    checkpoints_dir = 'exp3/'

    '''
    Display and storage parameters
    '''
    experiment_dir = 'exp3/'
    display_step = 10
    train_test = 'testing'

    evaluate_multiple_checkpoints(dataset, model, metrics, checkpoints_dir, batch_size=batch_size, device=device, 
                                  experiment_dir=experiment_dir, display_step=display_step, train_test=train_test)
    
