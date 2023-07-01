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

def train(dataset, model, model_opt, criterion, n_epochs=10, batch_size=10, device='cuda', experiment_dir='exp/', display_step=10):
    '''
    Training loop
    '''
    cur_step = 0
    losses = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input1, labels, input2 = next(iter(dataloader))

    for epoch in range(n_epochs):
        epoch_loss = 0
    # for input1, labels, input2 in tqdm.tqdm(dataloader):
        # Flatten the image
        input1, labels, input2 = input1.to(device), labels.to(device), input2.to(device)

        model_opt.zero_grad()
        pred = model(input1, input2)
        model_loss = criterion(pred, labels)
        model_loss.backward()
        model_opt.step()
        epoch_loss += model_loss.item()
        cur_step += 1

        losses.append(epoch_loss)

        '''
        Saves checkpoints, visualizes predictions and plots sosses
        '''
        if epoch % display_step == 0:

            # Saves snapshot of model's architecture
            print(f"Epoch {epoch}: Step {cur_step}: Model loss: {model_loss.item()}")

            # Visualizes predictions and ground truth
            visualize_batch(input1, labels, input2, pred, model, losses, epoch, experiment_dir, train_test='training')

            # Saves checkpoing with model's current state
            torch.save(model.state_dict(), experiment_dir + 'checkpoint' + str(epoch) + '.pth')




if __name__ == '__main__':
    '''
    Dataset parameters
    '''
    device = 'cuda:1'
    data_dir = '/data/farriaga/atd_12k/Line_Art/train_10k'
    # data_dir = 'mini_datasets/mini_train_triplets/'
    input_dim = 2
    label_dim = 1
    initial_shape = (512, 512)
    target_shape = (373, 373)
    binary_threshold = 0.75
    transform=transforms.Compose([transforms.ToTensor(),])
    dataset = MyDataset(data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold, crop_shape=target_shape)

    '''
    Training parameters
    '''
    model = unet_int.UNet(input_dim, label_dim).to(device)
    loss = nn.BCEWithLogitsLoss()
    lr = 0.0002
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 16
    num_epochs = 3000


    '''
    Visualization parameters
    '''
    display_step = 10
    experiment_dir = 'exp_overfit/'
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    train(dataset, model, opt, loss, n_epochs=num_epochs, batch_size=batch_size, device=device, experiment_dir=experiment_dir, display_step=display_step)

