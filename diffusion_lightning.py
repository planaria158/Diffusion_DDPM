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
from mlflow import log_metric, log_param
from tqdm import tqdm
import os
import torchvision
from torchvision.utils import make_grid
from unet_diffusion import UNet_Diffusion
from noise_scheduler import LinearNoiseScheduler


# -------------------------------------------------------------------
# Exponential moving average for more stable training
# copied from https://github.com/dome272/Diffusion-Models-pytorch
# -------------------------------------------------------------------
class EMA:
    def __init__(self, beta=0.9999, warmup=2000):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.warmup_steps = warmup

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model):
        if self.step < self.warmup_steps:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DDPM(LightningModule):
    def __init__(self, config, diffusion_config, **kwargs):
        super().__init__()

        self.criterion = nn.MSELoss()
        self.num_timesteps = diffusion_config['num_timesteps']
        self.beta_start = diffusion_config['beta_start']
        self.beta_end = diffusion_config['beta_end']
        self.num_samples = diffusion_config['num_samples']
        self.num_grid_rows = diffusion_config['num_grid_rows']
        self.sample_epochs = diffusion_config['sample_epochs']
        self.task_name = diffusion_config['task_name']

        self.loss_weighting = config['loss_weighting']
        self.img_size = tuple(config['img_size'])
        self.model = UNet_Diffusion(config)
        self.scheduler = LinearNoiseScheduler(self.num_timesteps, self.beta_start, self.beta_end)
        self.ema = EMA(beta=0.9999) 
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.save_hyperparameters()

        log_param('criterion', self.criterion)
        log_param('ema_warmup', self.ema.warmup_steps)
        log_param('ema_beta', self.ema.beta)

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
        # if self.loss_weighting == 'PP' :
        #     # PP-weighted MSE loss
        #     weights = self.scheduler.get_pp_weights(tstep, normalize=False).to(imgs)
        #     weights = weights/torch.max(weights)
        #     err2 = (noise - noise_pred)**2
        #     loss = err2.mean(dim=list(range(1, len(err2.shape))))
        #     loss = (loss * weights).mean()
        #     # loss = (weights * (noise - noise_pred)**2).mean() 
        # else :
        # Standard loss function (MSE)
        loss = self.criterion(noise_pred, noise)

        return loss
    
    # ---------------------------------------------------------------
    # Training step:
    # ---------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss = self.common_forward(batch)
        self.log_dict({"loss": loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # After every batch, apply the EMA-based weights update
        self.ema.step_ema(self.ema_model, self.model)
        return

    def on_train_epoch_end(self):
        # Generate images periodically during training
        if (self.current_epoch > 1) & (self.current_epoch % self.sample_epochs == 0):
            # only run on a single gpu
            if torch.cuda.current_device() == 0:
                self._sample()
        return
    
    # Generate a grid of diffusion images
    def _sample(self):
        # use self.ema_model since it's already set to eval, grad_not_required
        device = self.device
        task_name = self.task_name

        xt = torch.randn((self.num_samples, 3, self.img_size[0], self.img_size[1])).to(device)

        for i in tqdm(reversed(range(self.num_timesteps))):
            # Get prediction of noise
            noise_pred = self.ema_model(xt, torch.as_tensor(i).unsqueeze(0).to(device))            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = self.scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            
        # Save final predicted image Xo to file.
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=self.num_grid_rows)
        img = torchvision.transforms.ToPILImage()(grid)
        out_path = os.path.join(task_name, 'samples') 
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        img.save(os.path.join(out_path, 'x0_epoch_{}.png'.format(self.current_epoch)))
        img.close()
        return

    # ---------------------------------------------------------------
    # Validation step:
    # ---------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        val_loss = self.common_forward(batch)
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return val_loss
    
    def on_load_checkpoint(self, checkpoint):
        print("\nRestarting from checkpoint")
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema.step = checkpoint['global_step'] 
        print('on_load_checkpoint: calling self.ema.step:', self.ema.step)
        return

    def configure_optimizers(self):
        lr = 0.0001
        b1 = 0.5
        b2 = 0.999
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2))
        log_param('optimizer', optimizer)
        log_param('Adam_lr', lr)
        log_param('Adam_b1', b1)
        log_param('Adam_b2', b2)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        return [optimizer] #, [scheduler]

 