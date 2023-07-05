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

def train(tra_dataset, model, model_opt, criterion, test_dataset=None, n_epochs=10, batch_size=10, device='cuda', experiment_dir='exp/', display_step=10, my_dataset=None):
    '''
    Training loop
    '''
    cur_step = 0
    tr_losses = []
    test_losses = []
    dataloader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        print('Epoch: ' + str(epoch))
        model.train()       # Set the model to training mode
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

        tr_losses.append(epoch_loss/len(dataloader))


        '''
        Performs testing if specified
        '''
        if test_dataset is not None:
            # Free up unused memory before starting testing process
            torch.cuda.empty_cache()
            
            # Set the model to evaluation mode
            model.eval()
            
            # Evaluate the model on the test dataset
            with torch.no_grad():
                test_loss = test(test_dataset, model, criterion, epoch, batch_size=batch_size, device=device, experiment_dir=experiment_dir,
                                 display_step=display_step, train_test='testing')
                test_losses.append(test_loss)
            
            # Set the model back to training mode
            model.train()

        '''
        Saves checkpoints, visualizes predictions and plots losses
        '''
        if epoch % display_step == 0:

            # Save snapshot of model architecture``
            if epoch == 0:
                with open(experiment_dir + 'model_architecture.txt', 'w') as f:
                    print(model, file=f)


            '''
            Performs testing in MY dataset if specified
            '''
            if my_dataset is not None:
                # Free up unused memory before starting testing process
                torch.cuda.empty_cache()
                
                # Set the model to evaluation mode
                model.eval()
                
                # Evaluate the model on the MY dataset
                with torch.no_grad():
                    unused_loss = test(my_test_dataset, model, criterion, epoch, batch_size=batch_size, device=device,
                                       experiment_dir=experiment_dir,
                                       display_step=display_step,
                                       train_test='extra_testing')
                
                # Set the model back to training mode
                model.train()

            print(f"Epoch {epoch}: Step {cur_step}: Training loss: {model_loss.item()} Testing loss: {test_losses[-1]}")
            
            # Visualizes predictions and ground truth
            visualize_batch(input1, labels, input2, pred, epoch,
                            experiment_dir=experiment_dir,
                            train_losses=tr_losses,
                            test_losses=test_losses,
                            train_test='training')

            # Saves checkpoing with model's current state
            torch.save(model.state_dict(), experiment_dir + 'checkpoint' + str(epoch) + '.pth')

def test(dataset, model, criterion, epoch, batch_size=8, device='cuda:1', experiment_dir='exp/', display_step=10, train_test='testing'):
    '''
    Testing a single epoch
    '''
    epoch_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input1, labels, input2 = 0, 0, 0

    for input1, labels, input2 in tqdm.tqdm(dataloader):
        # Flatten the image
        input1, labels, input2 = input1.to(device), labels.to(device), input2.to(device)

        pred = model(input1, input2)
        model_loss = criterion(pred, labels)
        epoch_loss += model_loss.item()

    '''
    Saves checkpoints, visualizes predictions and plots losses
    '''
    if epoch % display_step == 0:
        # Visualizes predictions and ground truth
        visualize_batch(input1, labels, input2, pred, epoch, experiment_dir=experiment_dir, test_losses=epoch_loss, train_test=train_test)
        
        # Returns average loss
    return epoch_loss/len(dataloader)


if __name__ == '__main__':
    '''
    Dataset parameters
    '''
    device = 'cuda:1'
    train_data_dir = '/data/farriaga/atd_12k/Line_Art/train_10k/'
    input_dim = 2
    label_dim = 1
    initial_shape = (512, 512)
    target_shape = (373, 373)
    binary_threshold = 0.75
    transform=transforms.Compose([transforms.ToTensor(),])
    train_dataset = MyDataset(train_data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold, crop_shape=target_shape)
    test_data_dir = '/data/farriaga/atd_12k/Line_Art/test_2k_original/'
    test_dataset = MyDataset(test_data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold, crop_shape=target_shape)

    #My dataset
    my_test_data_dir = 'mini_datasets/mini_real_test_triplets/'
    my_test_dataset = MyDataset(my_test_data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold, crop_shape=target_shape)

    '''
    Training parameters
    '''
    model = unet_int.UNet(input_dim, label_dim).to(device)
    loss = nn.BCEWithLogitsLoss()
    lr = 0.0002
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 8
    num_epochs = 100


    '''
    Visualization parameters
    '''
    display_step = 1
    experiment_dir = 'exp4/'
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    train(train_dataset, model, opt, loss, n_epochs=num_epochs, batch_size=batch_size, device=device,
           experiment_dir=experiment_dir, display_step=display_step, test_dataset=test_dataset, my_dataset=my_test_dataset)

