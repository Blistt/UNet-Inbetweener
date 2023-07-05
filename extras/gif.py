from utils import create_gif
from PIL import Image

# Saves gifs of the predicted and ground truth triplets

# reads image using image pillow 

input1 = Image.open('02_00(1).png').convert('L')
labels = Image.open('/home/farriaga/VAEInterpolator/mini_datasets/mini_real_test_triplets/steins_gate_1/frame2.png').convert('L')
input2 = Image.open('02_02(1).png').convert('L')
preds = Image.open('02_01(1).png').convert('L')



# Gif for generated triplet
input1.save('results/pred_.gif', save_all=True, append_images=[preds, input2], duration=500, loop=0)
# Gif for ground truth triplet
input1.save('results/true_.gif', save_all=True, append_images=[labels, input2], duration=500, loop=0)