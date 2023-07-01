from torch.utils.data import Dataset
import os
import cv2
from utils import crop, pre_process
from matplotlib import pyplot as plt
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None, binarize_at=0.0, resize_to=(0,0), crop_shape=(0,0)):
        self.data = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
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

        input1 = triplet[2]
        label = triplet[0]
        input2 = triplet[1]

        if self.crop_shape != (0,0):
            label = crop(label, self.crop_shape)

        return input1, label, input2