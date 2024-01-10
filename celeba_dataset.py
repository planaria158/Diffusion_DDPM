
import os
import pandas as pd
import torch
import numpy as np
import random
from torchvision.io import read_image
from torch.utils.data import Dataset

class CelebA(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, limit_size=False, size_limit=20):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.noise_factor = 0.50
        filepaths = pd.Series(list(self.img_dir.glob(r'**/*.jpg')), name='Filepath', dtype = 'object').astype(str)
        self.images = pd.concat([filepaths], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        if limit_size:
            assert size_limit > 0
            self.images = self.images.iloc[0:size_limit]

        print('Dataset has', self.images.shape[0], 'rows')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        num_rows = self.images.shape[0]
        done = False
        # filter out anything that is not an rgb image (there are some b&w images in this dataset)
        while not done:
            img_path = self.images.iloc[idx]['Filepath']
            image = read_image(img_path).to(torch.float32)
            if image.shape[0] == 3:
                done = True 
            else:
                # print('\nFound B&W image, continuing to search, idx:', idx)
                idx = random.randint(0, num_rows-1)

                
        if self.transform:
            image = self.transform(image)
            # Normalize image [-1, 1]
            image = (image - 127.5)/127.5
            
        if self.target_transform:
            label = self.target_transform(label)

        # return image.float(), image.float()  
        return image.float(), image.float() 