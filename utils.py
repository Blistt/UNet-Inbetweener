import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from torchvision import transforms
import imageio
from PIL import Image




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


"""Pre-Processing"""
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

def normalize(imgs):
    '''
    Normalizes each image in a batch of images to the range 0-1
        imgs: a pytorch tensor of shape (batch size, channels, height, width)
    '''
    imgs_min = imgs.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
    imgs_max = imgs.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    imgs_range = imgs_max - imgs_min
    return (imgs - imgs_min) / imgs_range
    


def show_tensor_images(image_tensor, num_images=16, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())


def create_gif(input1, labels, input2, pred, experiment_dir, epoch):
    input1, input2 = crop(input1[0], pred.shape), crop(input2[0], pred.shape)
    pred = Image.fromarray(np.squeeze((pred[0].detach().cpu().numpy() * 255), axis=0))
    input1 = Image.fromarray(np.squeeze((input1.detach().cpu().numpy() * 255), axis=0))
    input2 = Image.fromarray(np.squeeze((input2.detach().cpu().numpy() * 255), axis=0))
    labels = Image.fromarray(np.squeeze((labels[0].detach().cpu().numpy() * 255), axis=0))
    # Gif for generated triplet
    input1.save(experiment_dir + 'triplet_' + str(epoch) + 'true_.gif', save_all=True, append_images=[labels, input2], duration=500, loop=0)
    # Gif for ground truth triplet
    input1.save(experiment_dir + 'triplet_' + str(epoch) + 'pred_.gif', save_all=True, append_images=[pred, input2], duration=500, loop=0)


def visualize_batch(input1, labels, input2, pred, epoch, experiment_dir='exp/', train_losses=None, test_losses=None, train_test='training', figsize=(20,10)):
        # Creates experiment directory if it doesn't exist
        experiment_dir = experiment_dir + train_test + '/'
        if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

        if train_losses is not None and test_losses is not None:
            # Plots training and testing losses in the same plot
            plt.figure()
            plt.plot(train_losses, label='Training')
            plt.plot(test_losses, label='Testing')
            plt.title("Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(experiment_dir + 'loss' + str(epoch) + '.png')

            # Plots generated images
            plt.figure(figsize=figsize) # Create a new figure for the subplots
            plt.subplot(1,2,1)
            show_tensor_images(labels, num_images=labels.shape[0])
            plt.title("True")
            plt.subplot(1,2,2)
            show_tensor_images(pred, num_images=pred.shape[0])
            plt.title("Generated")
            plt.savefig(experiment_dir + 'gens' + str(epoch) + '.png')

            # Saves gifs of the predicted and ground truth triplets
            create_gif(input1, labels, input2, pred, experiment_dir, epoch)
        
        if train_losses is None and test_losses is not None:
            # Plots generated images
            plt.figure(figsize=figsize) # Create a new figure for the subplots
            plt.subplot(1,2,1)
            show_tensor_images(labels)
            plt.title("True")
            plt.subplot(1,2,2)
            show_tensor_images(pred)
            plt.title("Generated")
            plt.savefig(experiment_dir + 'gens' + str(epoch) + '.png')

            # Saves gifs of the predicted and ground truth triplets
            create_gif(input1, labels, input2, pred, experiment_dir, epoch)

        if train_losses is not None and test_losses is None:
            # Plots training and testing losses in the same plot
            plt.figure()
            plt.plot(train_losses, label='Training')
            plt.title("Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(experiment_dir + 'loss' + str(epoch) + '.png')

            # Plots generated images
            plt.figure(figsize=figsize)

            # Plots generated images
            plt.figure(figsize=figsize) # Create a new figure for the subplots
            plt.subplot(1,2,1)
            show_tensor_images(labels)
            plt.title("True")
            plt.subplot(1,2,2)
            show_tensor_images(pred)
            plt.title("Generated")
            plt.savefig(experiment_dir + 'gens' + str(epoch) + '.png')

            # Saves gifs of the predicted and ground truth triplets
            create_gif(input1, labels, input2, pred, experiment_dir, epoch)