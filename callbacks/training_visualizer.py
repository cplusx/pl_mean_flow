import os
import pytorch_lightning as pl
import torch
import torchvision
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import wandb

def unnorm(x):
    '''convert from range [-1, 1] to [0, 1]'''
    return (x+1) / 2

def clip_image(x, min=0., max=1.):
    return torch.clamp(x, min=min, max=max)

def format_dtype_and_shape(x):
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        if len(x.shape) == 4 and x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)
        x = x.detach().cpu().numpy()
    return x

def tensor2image(x):
    x = x.float() # handle bf16
    '''convert 4D (b, dim, h, w) pytorch tensor to wandb Image class'''
    grid_img = torchvision.utils.make_grid(
        x, nrow=4
    ).permute(1, 2, 0).detach().cpu().numpy()
    img = wandb.Image(
        grid_img
    )
    return img

class DiffusionTrainingLogger(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger=None,
        max_num_images: int=16,
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.max_num_images = max_num_images

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            generated = outputs['generated'][:self.max_num_images] # this is B, h, w, c
            generated = tensor2image(clip_image(
                torch.from_numpy(generated).permute(0, 3, 1, 2)
            ))
            self.wandb_logger.experiment.log({
                'val/generated': generated,
            })
