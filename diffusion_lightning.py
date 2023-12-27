#--------------------------------------------------------------------
#
# DDPM Diffusion Model
# as a pytorch lightning module.
#
#--------------------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.core import LightningModule
import torchvision.utils as vutils
from numpy.random import random, choice
import pickle
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from numpy.random import choice
import pytorch_lightning as pl

from celeba_dataset import CelebA
from unet_diffusion import UNet_Diffusion
from noise_scheduler import LinearNoiseScheduler

class DDPM(LightningModule):
    def __init__(self,
                **kwargs):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.num_timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.time_emb_dim = 256
        self.num_epochs = 500
        self.model = UNet_Diffusion(self.time_emb_dim)
        self.scheduler = LinearNoiseScheduler(self.num_timesteps, self.beta_start, self.beta_end)
        self.save_hyperparameters()
    
    def forward(self, noisy_im, t):
        return self.model(noisy_im, t)
    
    def common_forward(self, batch):
        imgs = batch[0]
        # Random noise
        noise = torch.randn_like(imgs) 
        # Timestep
        tstep = torch.randint(0, self.num_timesteps, (imgs.shape[0],)) 
        # Add noise to images according to timestep
        noisy_imgs = self.scheduler.add_noise(imgs, noise, tstep).to(imgs)
        # Model tries to learn the noise that was added to im to make noise_im
        noise_pred = self.forward(noisy_imgs, tstep.to(imgs))
        # Loss is our predicted noise relative to actual noise
        loss = self.criterion(noise_pred, noise)
        return loss
    
    # ---------------------------------------------------------------
    # Training step:
    # ---------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss = self.common_forward(batch)
        self.log_dict({"loss": loss}, prog_bar=True, sync_dist=True)
        return loss

    # ---------------------------------------------------------------
    # Validation step:
    # ---------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        val_loss = self.common_forward(batch)
        self.log_dict({"val_loss": val_loss}, prog_bar=True, sync_dist=True)
        return val_loss
    

    def configure_optimizers(self):
        lr = 0.00002  # was 0.0002
        b1 = 0.5
        b2 = 0.999
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2))
        return opt

 