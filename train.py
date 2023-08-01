from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_class import MyDataset
import models.unet_crop as unet_crop
import models.unet_full as unet_full
import tqdm
import torch
from utils import visualize_batch
import os
from torch import nn
from test import test

def train(tra_dataset, model, model_opt, criterion, test_dataset=None, n_epochs=10, batch_size=10, device='cuda', 
          save_checkpoints=True, experiment_dir='exp/', display_step=10, my_dataset=None):
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
                    unused_loss = test(my_dataset, model, criterion, epoch, batch_size=batch_size, device=device,
                                       experiment_dir=experiment_dir,
                                       display_step=display_step,
                                       train_test='extra_testing')
                
                # Set the model back to training mode
                model.train()

            print(f"Epoch {epoch}: Step {cur_step}: Training loss: {model_loss.item()} Testing loss: {test_losses[-1]}")
            
            # Visualizes predictions and ground truth
            with torch.no_grad():
                pred = torch.sigmoid(pred)         # Gets the sigmoid activation of the predictions
            visualize_batch(input1, labels, input2, pred, epoch,
                            experiment_dir=experiment_dir,
                            train_losses=tr_losses,
                            test_losses=test_losses,
                            train_test='training')

            # Saves checkpoing with model's current state
            if save_checkpoints:
                torch.save(model.state_dict(), experiment_dir + 'checkpoint' + str(epoch) + '.pth')