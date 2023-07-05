"""U-Net_Interpolator


# U-Net

**Supporting functions**
"""

import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')

def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels.
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    padding_y = (image.shape[-2]-new_shape[-2]) // 2
    padding_x = (image.shape[-1]-new_shape[-1]) // 2
    odd_y, odd_x = (image.shape[-2]-new_shape[-2]) % 2, (image.shape[-1]-new_shape[-1]) % 2
    # Crops whole batch or a single image
    if image.dim() > 3:
      cropped_image = image[:, :, padding_y+odd_y:image.shape[-2]-padding_y, padding_x+odd_x:image.shape[-1]-padding_x]
    else:
      cropped_image = image[:, padding_y+odd_y:image.shape[-2]-padding_y, padding_x+odd_x:image.shape[-1]-padding_x]
    return cropped_image

"""**Model**"""

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ContractingBlock, self).__init__()
        # You want to double the number of channels in the first convolution
        # and keep the same number of channels in the second.
        #### START CODE HERE ####
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3)
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #### END CODE HERE ####

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

    # Required for grading
    def get_self(self):
        return self

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=3, stride=1)
        self.activation = nn.ReLU()

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        return x

    # Required for grading
    def get_self(self):
        return self

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a UNet -
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, i1, i2=None):
        '''
        Function for completing a forward pass of FeatureMapBlock:
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        if i2 is not None:
          x = self.conv(torch.cat((i1, i2), dim=1))
        else:
          x = self.conv(i1)
        return x

"""## U-Net

Now you can put it all together! Here, you'll write a `UNet` class which will combine a series of the three kinds of blocks you've implemented.
"""

class UNet(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(UNet, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.expand1 = ExpandingBlock(hidden_channels * 16)
        self.expand2 = ExpandingBlock(hidden_channels * 8)
        self.expand3 = ExpandingBlock(hidden_channels * 4)
        self.expand4 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, i1, i2):
        '''
        Function for completing a forward pass of UNet:
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        # Keep in mind that the expand function takes two inputs,
        # both with the same number of channels.
        #### START CODE HERE ####
        x0 = self.upfeature(i1, i2)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.expand1(x4, x3)
        x6 = self.expand2(x5, x2)
        x7 = self.expand3(x6, x1)
        x8 = self.expand4(x7, x0)
        xn = self.downfeature(x8)
        #### END CODE HERE ####
        return xn

"""Pre-Processing"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def pre_process(img, binarize_at=0.0, resize_to=(0,0)):
    if binarize_at > 0.0:
        thresh = int(binarize_at * np.max(img))
        # Replace all values in img that are less than the max value with 0
        img[img<thresh] = 0
        img[img>=thresh] = 255

    if resize_to != (0,0):
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_CUBIC)

    if binarize_at > 0.0:
        thresh = int(binarize_at * np.max(img))
        # Replace all values in img that are less than the max value with 0
        img[img<thresh] = 0
        img[img>=thresh] = 255


    return img

"""Dataset Loading"""
from torch.utils.data import Dataset
import os
import cv2

class MyDataset(Dataset):
    def __init__(self, data, transform=None, binarize_at=0.0, resize_to=(94,94), crop_shape=(0,0)):
        self.data = data
        self.transform = transform
        self.binarize_at = binarize_at
        self.resize_to = resize_to
        self.crop_shape = crop_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        triplet_path = self.data[idx]
        #Extracts all three frames of the triplet
        triplet = []
        for img_path in os.listdir(triplet_path):
            img = cv2.imread(os.path.join(triplet_path, img_path), cv2.IMREAD_GRAYSCALE)
            if self.binarize_at > 0.0 or self.resize_to != (0,0):
                img = pre_process(img, binarize_at=self.binarize_at, resize_to=self.resize_to)
            if self.transform:
                img = self.transform(img)
            triplet.append(img)
        if self.crop_shape != (0,0):
            triplet[1] = crop(triplet[1], self.crop_shape)
        return triplet[0], triplet[1], triplet[2]


import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import os

# Dataset parameters
data_dir = '/data/farriaga/atd_12k/Line_Art/train_10k'
triplet_paths = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
transform=transforms.Compose([transforms.ToTensor(),])
binarize = 0.75

# Training parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
input_dim = 2
label_dim = 1
display_step = 10
batch_size = 16
lr = 0.0002
initial_shape = 512
target_shape = 373
device = 'cuda'

dataset = MyDataset(triplet_paths, transform=transform, binarize_at=binarize, resize_to=(initial_shape, initial_shape),
                       crop_shape=(target_shape, target_shape))

def train():
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    unet = UNet(input_dim, label_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    cur_step = 0
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        for input1, labels, input2 in tqdm(dataloader):
            # Flatten the image
            input1, labels, input2 = input1.to(device), labels.to(device), input2.to(device)

            unet_opt.zero_grad()
            pred = unet(input1, input2)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()
            epoch_loss += unet_loss.item()
            cur_step += 1

        losses.append(epoch_loss)
       
            # Plot the list of losses
        if epoch % display_step == 0:
            print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
            path = 'graphs/'
            plt.figure()
            plt.plot(losses)
            plt.title("Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(path + 'loss' + str(epoch) + '.png')
            
            plt.figure() # Create a new figure for the subplots
            plt.subplot(1,2,1)
            show_tensor_images(labels)
            plt.title("True")
            plt.subplot(1,2,2)
            show_tensor_images(pred)
            plt.title("Generated")
            plt.savefig(path + 'recon' + str(epoch) + '.png')
            # Save checkpoints
            torch.save(unet.state_dict(), 'checkpoints/unet/baseline_unet_' + str(epoch) + '.pth')
        
        
train()