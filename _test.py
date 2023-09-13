from torch.utils.data import DataLoader
import tqdm
from torch import nn
import torch
from _loss import get_gen_loss
from torchvision.utils import save_image
from _utils.utils import create_gif, write_log
from collections import defaultdict
import numpy as np

# testing function
def test(dataset, gen, epoch, results_batch=None, display_step=10, plot_step=10, r1=nn.BCELoss(), 
         lambr1=0.5, r2=None, r3=None, lambr2=None, lambr3=None, metrics=None, batch_size=12, device='cuda', 
         experiment_dir='exp/'):
    '''
    Tests a single epoch
    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gen_epoch_loss = []
    disc_epoch_loss = []
    step_num = 0
    display_step = len(dataloader)//display_step
    results_e = defaultdict(list)     # Stores internal metrics for an epoch
    
    for input1, real, input2 in tqdm.tqdm(dataloader):
        input1, real, input2 = input1.to(device), real.to(device), input2.to(device)

        preds = gen(input1, input2)

        gen_loss = get_gen_loss(preds, real, r1=r1, lambr1=lambr1, r2=r2, r3=r3, lambr2=lambr2, lambr3=lambr3, device=device)

        '''Train discriminator'''        
        gen_epoch_loss.append(gen_loss.item())

        '''Compute evaluation metrics'''
        if metrics is not None:
            # Transfer tensors to other device to avoid issues with memory leak
            other_device = 'cuda:1' if device == 'cuda:0' else 'cuda:0'
            preds = preds.to(other_device)
            real = real.to(other_device)
            
            # Compute metrics
            raw_metrics = metrics(preds, real)
            for k, v in raw_metrics.items():
                results_batch[k].append(v.item())
                results_e[k].append(v.item())
            
            print('number of overall batch losses', len(results_batch['ssim']))
            print('number of batch losses in epoch', len(results_e['ssim']))

        if step_num % display_step == 0 and epoch % plot_step == 0:
            # Saves torch image with the batch of predicted and real images
            id = str(epoch) + '_' + str(step_num)
            save_image(real, experiment_dir + 'batch_' + id + '_real.png', nrow=4, normalize=True)
            save_image(preds, experiment_dir + 'batch_' + id + '_preds.png', nrow=4, normalize=True)
            create_gif(input1, real, input2, preds, experiment_dir, id) # Saves gifs of the predicted and ground truth triplets

        step_num += 1
            
    return gen_epoch_loss, disc_epoch_loss, results_e, results_batch
