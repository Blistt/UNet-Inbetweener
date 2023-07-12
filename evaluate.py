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
from utils import normalize
from torchvision.utils import save_image



def evaluate(dataset, model, metrics, batch_size=8, device='cuda:1', experiment_dir='exp/', display_step=10, train_test='testing'):
    '''
    Evaluates dataset on the given set of metrics
    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input1, labels, input2 = 0, 0, 0

    results = defaultdict(list)
    for input1, labels, input2 in tqdm.tqdm(dataloader):
        # Flattens the image
        input1, labels, input2 = input1.to(device), labels.to(device), input2.to(device)

        with torch.no_grad():
            pred = model(input1, input2)
            pred = torch.sigmoid(pred)
            save_image(pred, 'pred_torchvision_raw.png', normalize=False)
            save_image(pred, 'pred_torchvision.png', normalize=True)
            raw_metrics = metrics(pred, labels)
        for k,v in raw_metrics.items():
            results[k].append(v.item())
    
    # Returns the average of all metrics
    return {k: np.mean(results[k]) for k,v in results.items()}


# Main function
if __name__ == '__main__':
    device = torch.device('cuda')

    '''
    Dataset parameters
    '''
    data_dir = 'mini_datasets/eval/'
    input_dim = 2
    label_dim = 1
    initial_shape = (512, 512)
    target_shape = (373, 373)
    binary_threshold = 0.75
    dataset = MyDataset(data_dir, transform=transforms.ToTensor(), resize_to=initial_shape, binarize_at=binary_threshold,
                         crop_shape=target_shape)
    batch_size = 8

    '''
    Model parameters
    '''
    model = unet_int.UNet(input_dim, label_dim).to(device)
    model.load_state_dict(torch.load('exp3/checkpoint10.pth'))
    model = model.eval()
    metrics = torchmetrics.MetricCollection({
        'psnr': my_metrics.PSNRMetricCPU(),
        'ssim': my_metrics.SSIMMetricCPU(),
        'chamfer': chamfer_dist.ChamferDistance2dMetric(binary=0.5),
        'mse': torchmetrics.MeanSquaredError(),
    }).to(device).eval()

    results = evaluate(dataset, model, metrics, batch_size=batch_size, device=device)
    print(results)

