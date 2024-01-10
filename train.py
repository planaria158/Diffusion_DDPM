
# %%
import os
import torch
import yaml
import mlflow
from mlflow import log_metric, log_artifact, log_params, log_param
import argparse
from torch import utils
from torch import nn
import pytorch_lightning as pl
from torchvision.transforms.v2 import Resize, Compose, ToDtype, RandomHorizontalFlip 

from celeba_dataset import CelebA
from diffusion_lightning import DDPM

# mlflow. set_tracking_uri() 

# %%
def train(args):
    #--------------------------------------------------------------------
    # Read config
    #--------------------------------------------------------------------
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)
    log_params(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

# %%
    #--------------------------------------------------------------------
    # Dataset, Dataloader
    #--------------------------------------------------------------------
    from pathlib import Path
    image_dir_train = Path(dataset_config['train_path'])
    image_dir_valid = Path(dataset_config['valid_path'])

    img_size = tuple(model_config['img_size'])
    batch_size = train_config['batch_size']

    train_transforms = Compose([ToDtype(torch.float32, scale=False),
                                RandomHorizontalFlip(p=0.50),
                                Resize(img_size, antialias=True)
                                ])

    train_dataset = CelebA(image_dir_train, transform=train_transforms)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, persistent_workers=True)

# %%
    #--------------------------------------------------------------------
    # Lightning module
    #--------------------------------------------------------------------
    if train_config['restart']:
        print('Restarting from checkpoint')
        path = os.path.join(train_config['log_dir'], train_config['checkpoint_name'])
        model = DDPM.load_from_checkpoint(checkpoint_path=path)
    else:
        print('Starting from new model instance')
        model = DDPM(model_config, diffusion_config)

    total_params = sum(param.numel() for param in model.parameters())
    print('Model has:', int(total_params//1e6), 'M parameters')
    log_param('model_parameter_count', total_params)

# %%
    #--------------------------------------------------------------------
    # Training
    #--------------------------------------------------------------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=train_config['save_top_k'],
        every_n_epochs=train_config['checkpoint_every_n_epochs'],
        monitor = train_config['monitor'],
        mode = train_config['mode']
    )

    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=train_config['log_dir'], default_hp_metric=False)

    # Note: I tried to run in single and mixed-precision; both produced NANs.
    if train_config['accelerator'] == 'gpu':
        trainer = pl.Trainer(strategy='ddp_find_unused_parameters_true', 
                             accelerator=train_config['accelerator'], 
                             devices=train_config['devices'], 
                             max_epochs=train_config['num_epochs'], 
                             logger=logger, 
                             log_every_n_steps=train_config['log_every_nsteps'], 
                             callbacks=[checkpoint_callback]) 
    else:
        trainer = pl.Trainer(accelerator=train_config['accelerator'], 
                             max_epochs=train_config['num_epochs'], 
                             logger=logger, 
                             log_every_n_steps=train_config['log_every_nsteps'], 
                             callbacks=[checkpoint_callback]) 


    trainer.fit(model=model, train_dataloaders=train_loader) 

    print('\n\nTraining complete!')

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)

