from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
import torch
from utils import visualize_batch


def test(dataset, model, criterion, epoch, batch_size=8, device='cuda:1', experiment_dir='exp/', display_step=10, train_test='testing'):
    '''
    Testing a single epoch
    '''
    epoch_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input1, labels, input2 = 0, 0, 0

    for input1, labels, input2 in tqdm.tqdm(dataloader):
        with torch.no_grad():
            model.eval()
            # Flatten the image
            input1, labels, input2 = input1.to(device), labels.to(device), input2.to(device)
            pred = model(input1, input2)
            model_loss = criterion(pred, labels)
            epoch_loss += model_loss.item()


    '''
    Saves checkpoints, visualizes predictions and plots losses
    '''
    if epoch % display_step == 0:
    
        # Binarizes prediction
        with torch.no_grad():
            pred = (torch.sigmoid(pred) > 0.5).float()
            # Visualizes predictions and ground truth
            visualize_batch(input1, labels, input2, pred, epoch, experiment_dir=experiment_dir, test_losses=epoch_loss, train_test=train_test)
        
        # Returns average loss
    return epoch_loss/len(dataloader)
