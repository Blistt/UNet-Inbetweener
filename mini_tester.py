from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_class import MyDataset
import unet_int
import tqdm
from utils import show_tensor_images, create_gif
import torch

'''
 Test In-between generation in the miniset
 '''
def test(dataset, model, results_path, batch_size=1, exp_no=0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_count = 0
    for input1, reals, input2 in tqdm.tqdm(dataloader):
        # Flatten the image
        input1, reals, input2 = input1.to(device), reals.to(device), input2.to(device)
        # Generate in-betweens
        preds = model(input1, input2)
        plt.subplot(1,2,1)
        show_tensor_images(reals, num_images=1, size=(1, 28, 28))
        plt.title("True")
        plt.subplot(1,2,2)
        show_tensor_images(preds, num_images=1, size=(1, 28, 28))
        plt.title("Generated")
        plt.savefig(results_path + 'inter' + str(batch_count) + str(exp_no) + '.png')
        batch_count += 1
        
        
        # Saves gifs of the predicted and ground truth triplets
        create_gif(input1, reals, input2, preds, results_path, exp_no)

if __name__ == '__main__':
    '''
    Dataset specifications
    '''
    device = 'cuda:0'
    data_dir = 'mini_datasets/single_triplet/'
    input_dim = 2
    label_dim = 1
    results_path = 'results/'

    '''
    Input parameters
    '''
    initial_shape = (512, 512)
    target_shape = (373, 373)
    binary_threshold = 0.75
    transform=transforms.Compose([transforms.ToTensor(),])
    batch_size = 1

    dataset = MyDataset(data_dir, transform=transform, resize_to=initial_shape, binarize_at=binary_threshold, crop_shape=target_shape)

    '''
    Model parameters
    '''
    model = unet_int.UNet(input_dim, label_dim).to(device)
    checkpoint = 'Experiments/exp0/baselines_100epochs.pth'

    # Load saved weights
    weights = torch.load(checkpoint)
    model.load_state_dict(weights)

    test(dataset, model, results_path, batch_size=batch_size, exp_no=4)