
import os
import torch
from torch import utils
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.transforms.v2 import Resize, Compose, ToDtype, RandomHorizontalFlip, RandomVerticalFlip 
from torchvision.transforms.v2 import RandomResizedCrop, RandomRotation, GaussianBlur, RandomErasing

from celeba_dataset import CelebA
from diffusion_lightning import DDPM

#--------------------------------------------------------------------
# Dataset, Dataloader
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# Dataset, Dataloader
#--------------------------------------------------------------------
from pathlib import Path
image_dir_train = Path('../data/img_align_celeba/img_align_celeba/train/')
image_dir_valid = Path('../data/img_align_celeba/img_align_celeba/valid/')

img_size = (64,64) 
batch_size = 10


train_transforms = Compose([ToDtype(torch.float32, scale=False),
                            RandomHorizontalFlip(p=0.50),
                            # RandomVerticalFlip(p=0.25),
                            # transforms.RandomApply(nn.ModuleList([GaussianBlur(kernel_size=7)]), p=0.5),
                            # transforms.RandomApply(nn.ModuleList([RandomRotation(10.0)]), p=0.5),
                            # RandomResizedCrop(size=img_size, scale=(0.3, 1.0), antialias=True),
                            # RandomErasing(p=0.5, scale=(0.02, 0.20)),
                            Resize(img_size, antialias=True)
                            ])

valid_transforms = Compose([ToDtype(torch.float32, scale=False),
                            Resize(img_size, antialias=True)
                            ])

train_dataset = CelebA(image_dir_train, transform=train_transforms)
train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=5, persistent_workers=True)

# valid_dataset = CelebA(image_dir_valid, transform=valid_transforms)
# valid_loader = utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle = False, num_workers=5, persistent_workers=True)

#--------------------------------------------------------------------
# Lightning module
#--------------------------------------------------------------------
model = DDPM()
# model = DDPM.load_from_checkpoint(checkpoint_path='/home/mark/dev/diffusion/lightning_logs/version_8/checkpoints/epoch=4-step=75975.ckpt') 

total_params = sum(param.numel() for param in model.parameters())
print('Model has:', int(total_params//1e6), 'M parameters')

#--------------------------------------------------------------------
# Training
#--------------------------------------------------------------------
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=10,
    every_n_epochs=1,
    monitor = 'loss',
    mode = 'min'
)

from lightning.pytorch.loggers import TensorBoardLogger
logger = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs", default_hp_metric=False)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=500,
                     logger=logger, log_every_n_steps=1000, callbacks=[checkpoint_callback]) 

trainer.fit(model=model, train_dataloaders=train_loader) #, val_dataloaders=valid_loader) 



