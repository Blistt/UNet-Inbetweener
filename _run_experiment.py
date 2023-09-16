import torch
from torch import nn
from torchvision import transforms
from _generator_crop import UNetCrop
from _generator_light import GeneratorLight
from _utils.utils import weights_init, siren_weights_init
from _dataset_class import MyDataset
import os
import torchmetrics
import _eval.my_metrics as my_metrics
import _eval.chamfer_dist as chamfer_dist
from _loss import GDL, MS_SSIM, EDT_Loss
from _train import train
import time
import argparse
import configparser
import ast

# Argumment parser for configuration file
def parse_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config['DEFAULT']

if __name__ == '__main__':

    ### Configuration file parser ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.ini', help='Path to configuration file')
    args = parser.parse_args()
    config = parse_config(args.config)
    ###   ----------------------   ###


    device = config.get('device')                               # Device to use for training

    ''' -------------------------------------- Loss function parameters --------------------------------------'''
    r1 = eval(config.get('r1'))                                 # Reconstruction loss 1
    r2 = eval(config.get('r2'))                                 # Reconstruction loss 2
    r3 = eval(config.get('r3'))                                 # Reconstruction loss 3
    r1_lambda = config.getfloat('r1_lambda')                    # Reconstruction loss 1 weight        
    r2_lambda = config.getfloat('r2_lambda')                    # Reconstruction loss 2 weight
    r3_lambda = config.getfloat('r3_lambda')                    # Reconstruction loss 3 weight


    '''-------------------------------------- Training loop parameters --------------------------------------'''
    n_epochs = config.getint('n_epochs')                        # Number of epochs
    input_dim = config.getint('input_dim')                      # Input channels (1 for each grayscale input frame)
    label_dim = config.getint('label_dim')                      # Output channels (1 for each grayscale output frame)
    hidden_channels = config.getint('hidden_channels')          # Hidden channels of the generator and discriminator
    batch_size = config.getint('batch_size')                    # Batch size
    lr = config.getfloat('lr')                                  # Learning rate
    b1 = config.getfloat('b1')                                  # Adam: decay of first order momentum of gradient
    b2 = config.getfloat('b2')                                  # Adam: decay of second order momentum of gradient
    img_size = ast.literal_eval(config.get('img_size'))         # Frames' image size - (width, height)
    target_size = ast.literal_eval(config.get('target_size'))   # Cropped frames' image size - (width, height)
    overfit_batch = config.getboolean('overfit_batch')          # Whether to overfit a single batch or not            


    '''-------------------------------------- Model --------------------------------------'''
    gen = eval(config.get('gen'))                               # Generator
    gen_opt = eval(config.get('gen_opt'))                       # Generator optimizer  
    save_checkpoints = config.getboolean('save_checkpoints')    # Whether to save checkpoints or not


    '''-------------------------------------- Dataset parameters --------------------------------------'''
    transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.Resize(img_size, antialias=True),])
    binary_threshold = config.getfloat('binary_threshold')      # Threshold for binarizing input images
    # Training dataset
    train_data_dir = config.get('train_data_dir')
    train_dataset = MyDataset(train_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold,
                               crop_shape=target_size)
    # Testing dataset (optional)
    val_data_dir = config.get('val_data_dir')
    val_dataset = MyDataset(val_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold,
                             crop_shape=target_size)
    # MY dataset (optional)
    test_data_dir = config.get('test_data_dir')
    test_dataset = MyDataset(test_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold,
                           crop_shape=target_size)
    

    '''-------------------------------------- Evaluation Metrics --------------------------------------'''
    other_device = 'cuda:1' if device == 'cuda:0' else 'cuda:0'
    metrics = torchmetrics.MetricCollection({
        'psnr': my_metrics.PSNRMetricCPU(),
        'ssim': my_metrics.SSIMMetricCPU(),
        'chamfer': chamfer_dist.ChamferDistance2dMetric(binary=0.5),
        'mse': torchmetrics.MeanSquaredError(),
    }).to(other_device).eval()


    '''-------------------------------------- Visualization parameters --------------------------------------'''
    display_step = config.getint('display_step')                # How many times per epoch to display/visualize the images
    plot_step = config.getint('plot_step')                      # How many times per epoch to plot the loss
    experiment_dir = '/data/farriaga/Experiments/' + os.path.splitext(os.path.basename(args.config))[0] + '/'
    print('config_file: ', os.path.splitext(os.path.basename(args.config))[0] + '/')
    print('experiment_dir: ', experiment_dir)
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)


    '''-------------------------------------- Model Loading parameters --------------------------------------'''
    pretrain = config.get('pretrain')                           # 'pretrain', 'load' or 'none'
    pre_train_epochs = 100

    # Loads pre-trained model if specified
    if pretrain=='load':
        loaded_state = torch.load('/data/farriaga/Experiments/unet_int/exp3/checkpoint30.pth')
        gen.load_state_dict(loaded_state)
    else:
        gen = gen.apply(weights_init)

    '''
    Miniature experiment parameters - Overfiting a single batch
    '''
    if overfit_batch == True:
        train_data_dir = 'mini_datasets/mini_train_triplets/'
        train_dataset = MyDataset(train_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold,
                               crop_shape=target_size)        
        val_data_dir = 'mini_datasets/mini_test_triplets/'
        val_dataset = MyDataset(val_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold,
                                crop_shape=target_size)
        n_epochs = min(1000, n_epochs * 100)
        display_step = 1
        plot_step = n_epochs // 50                              # Ensures there is always 50 plots
        print('plot_step: ', plot_step)


    '''-------------------------------------- Execute Experiment --------------------------------------'''
    # Records time it takes to train the model
    start_time = time.time()

    train(train_dataset, gen, gen_opt, r1=r1, lambr1=r1_lambda, r2=r2, r3=r3, lambr2=r2_lambda, lambr3=r3_lambda, 
          n_epochs=n_epochs, batch_size=batch_size, device=device, metrics=metrics, display_step=display_step, 
          plot_step=plot_step, val_dataset=val_dataset, test_dataset=test_dataset, save_checkpoints=save_checkpoints, 
          experiment_dir=experiment_dir)
    
    # Saves the time the experiment took in a text file in experiment directory
    with open(experiment_dir + 'time.txt', 'w') as f:
        f.write(f'Training took {(time.time() - start_time)/60} minutes')