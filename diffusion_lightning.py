#--------------------------------------------------------------------
#
# DDPM Diffusion Model
# as a pytorch lightning module.
#
#--------------------------------------------------------------------

import torch
from pytorch_lightning.core import LightningModule
from torch import nn
import pytorch_lightning as pl
import copy

from unet_diffusion import UNet_Diffusion
from noise_scheduler import LinearNoiseScheduler


# -------------------------------------------------------------------
# Exponential moving average for more stable training
# copied from https://github.com/dome272/Diffusion-Models-pytorch
# -------------------------------------------------------------------
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



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
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

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
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Apply the EMA-based weights update
        self.ema.step_ema(self.ema_model, self.model)
        return


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

 