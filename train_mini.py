from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_class import MyDataset
import unet_int
import tqdm
import torch
from utils import visualize_batch
import os
from torch import nn

def train(tra_dataset, model, model_opt, criterion, test_dataset=None, n_epochs=10, batch_size=10, device='cuda', experiment_dir='exp/', display_step=10):
    '''
    Training loop
    '''
    cur_step = 0
    tr_losses = []
    test_losses = []
    dataloader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        epoch_loss = 0
        for input1, labels, input2 in tqdm.tqdm(dataloader):
            # Flatten the image
            input1, labels, input2 = input1.to(device), labels.to(device), input2.to(device)

            model_opt.zero_grad()
            pred = model(input1, input2)
            model_loss = criterion(pred, labels)
            model_loss.backward()
            model_opt.step()
            epoch_loss += model_loss.item()
            cur_step += 1

        tr_losses.append(epoch_loss/len(epoch_loss))

        '''
        Saves checkpoints, visualizes predictions and plots losses
        '''
        if epoch % display_step == 0:

            if epoch == 0:
                # Save snapshot of model architecture``
                with open(experiment_dir + 'model_architecture.txt', 'w') as f:
                    print(model, file=f)

            if test_dataset is not None:
                test_losses.append(test(test_dataset, model, criterion, epoch, batch_size=batch_size, device=device, experiment_dir=experiment_dir, display_step=display_step))
                

            print(f"Epoch {epoch}: Step {cur_step}: Training loss: {model_loss.item()} Testing loss: {test_losses[-1]}")
            # Visualizes predictions and ground truth
            visualize_batch(input1, labels, input2, pred, epoch, experiment_dir=experiment_dir, train_losses=tr_losses, test_losses=test_losses, train_test='training')

            # Saves checkpoing with model's current state
            torch.save(model.state_dict(), experiment_dir + 'checkpoint' + str(epoch) + '.pth')

def test(dataset, model, criterion, epoch, batch_size=8, device='cuda:1', experiment_dir='exp/', display_step=10):
    '''
    Testing loop
    '''
    cur_step = 0
    test_losses = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for input1, labels, input2 in tqdm.tqdm(dataloader):
        # Flatten the image
        input1, labels, input2 = input1.to(device), labels.to(device), input2.to(device)

        pred = model(input1, input2)
        model_loss = criterion(pred, labels)
        epoch_loss += model_loss.item()
        cur_step += 1

        '''
        Saves checkpoints, visualizes predictions and plots losses
        '''
        if cur_step % display_step == 0:
            # Visualizes predictions and ground truth
            visualize_batch(input1, labels, input2, pred, epoch, experiment_dir=experiment_dir, test_losses=losses, train_test='testing')
        
        
        
        # Returns average loss
    return epoch_loss/len(epoch_loss)


if __name__ == '__main__':
    '''
    Dataset parameters
    '''
    device = 'cuda:0'
    train_data_dir = 'mini_datasets/mini_train_triplets/'
    input_dim = 2
    label_dim = 1
    initial_shape = (512, 512)
    target_shape = (373, 373)
    binary_threshold = 0.75
    transform=transforms.Compose([transforms.ToTensor(),])
    train_dataset = MyDataset(train_data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold, crop_shape=target_shape)
    test_data_dir = 'mini_datasets/mini_test_triplets/'
    test_dataset = MyDataset(test_data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold, crop_shape=target_shape)

    '''
    Training parameters
    '''
    model = unet_int.UNet(input_dim, label_dim).to(device)
    loss = nn.BCEWithLogitsLoss()
    lr = 0.0002
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 2
    num_epochs = 100


    '''
    Visualization parameters
    '''
    display_step = 10
    experiment_dir = 'exp_overfit/'
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    train(train_dataset, model, opt, loss, test_dataset=test_dataset, n_epochs=num_epochs, batch_size=batch_size, device=device, experiment_dir=experiment_dir, display_step=display_step)

