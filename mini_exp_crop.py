from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_class import MyDataset
import models.unet_crop as unet_crop
import models.unet_full as unet_full
import torch
from utils import visualize_batch
import os
from torch import nn
from train import train, test

if __name__ == '__main__':
    '''
    Dataset parameters
    '''
    device = 'cuda:1'
    # train_data_dir = '/data/farriaga/atd_12k/Line_Art/train_10k/'
    train_data_dir = 'mini_datasets/mini_train_triplets/'
    input_dim = 2
    label_dim = 1
    initial_shape = (512, 512)
    target_shape = (373, 373)
    binary_threshold = 0.75
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Grayscale(num_output_channels=1),
                                  transforms.Resize(initial_shape),])
    train_dataset = MyDataset(train_data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold,
                              crop_shape=target_shape)
    # test_data_dir = '/data/farriaga/atd_12k/Line_Art/test_2k_original/'
    test_data_dir = 'mini_datasets/mini_test_triplets/'
    test_dataset = MyDataset(test_data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold,
                             crop_shape=target_shape)

    #My dataset
    my_test_data_dir = 'mini_datasets/mini_real_test_triplets/'
    my_test_dataset = MyDataset(my_test_data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold,
                                crop_shape=target_shape)

    '''
    Training parameters
    '''
    model = unet_crop.UNet(input_dim, label_dim).to(device)
    loss = nn.BCEWithLogitsLoss()
    lr = 0.0002
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 12
    num_epochs = 1000


    '''
    Visualization parameters
    '''
    display_step = 20
    experiment_dir = 'temp_test/'
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    train(train_dataset, model, opt, loss, n_epochs=num_epochs, batch_size=batch_size, device=device,
           experiment_dir=experiment_dir, display_step=display_step, test_dataset=test_dataset, my_dataset=my_test_dataset)

