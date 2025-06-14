import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import pytorch_lightning as pl
from einops import rearrange
from functools import partial
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor

class MeanFlowTrainer(pl.LightningModule):
    def __init__(
        self,
        pipe, 
        guidance_scale: float=2.0,
        optim_args: dict={},
        loss_weights: dict={
            'l2': 1.0,
        },
        gradient_checkpointing: bool=False,
        use_8bit_adam: bool=True,
        accumulate_grad_batches: int=8,
        use_ema: bool=False,
        ema_decay: float=0.99,
        ema_start: int=100,
        t_r_equal_rate: float=0.25, # according to paper, 75% t != r
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pipe = pipe
        self.denoiser = pipe.denoiser
        self.guidance_scale = guidance_scale
        self.optim_args = optim_args
        self.loss_weights = loss_weights
        self.use_8bit_adam = use_8bit_adam
        self.accumulate_grad_batches = accumulate_grad_batches
        self.t_r_equal_rate = t_r_equal_rate
        if self.accumulate_grad_batches > 1:
            self.automatic_optimization = False
        if use_ema:
            self.use_ema = use_ema
            self.ema_decay = ema_decay
            self.ema_start = ema_start
            self.init_ema_model()

        self.call_save_hyperparameters()

        if gradient_checkpointing:
            if self.denoiser._supports_gradient_checkpointing:
                self.denoiser.enable_gradient_checkpointing()

        for name, m in self.denoiser.named_modules():
            if isinstance(m, Attention):
                m.set_processor(AttnProcessor()) # compatible with jvp

    def init_ema_model(self):
        if self.ema_decay:
            self.ema_denoiser = AveragedModel(self.denoiser, multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay))
            if self.local_rank == 0:
                print('INFO: EMA model enabled with decay', self.ema_decay)
            self.denoiser_state_dict = None

    def call_save_hyperparameters(self):
        self.save_hyperparameters(
            ignore=['ema_denoiser', 'denoiser', 'pipe']
        )

    def sample_t_and_r(self, batch_size):
        lognormal_sampler = torch.distributions.log_normal.LogNormal(loc=torch.tensor([-2.0]), scale=torch.tensor([2.0]))
        t = lognormal_sampler.sample((batch_size,)).clamp(min=0.01, max=1.0).to(self.denoiser.device).squeeze()
        r = lognormal_sampler.sample((batch_size,)).clamp(min=0.0, max=1.0).to(self.denoiser.device).squeeze()
        # ensure t >= r
        t_ = torch.where(t < r, r, t)
        r_ = torch.where(t < r, t, r)
        t, r = t_, r_
        r = torch.where(
            torch.rand(batch_size, device=self.denoiser.device) < self.t_r_equal_rate,  # with probability t_r_equal_rate
            t,  # if r is less than t_r_equal_rate, set r = t
            r  # otherwise keep r as is
        )
        return t, r        

    def train_internal_step(self, batch, batch_idx, mode='train'):
        image = batch['image']
        class_labels = batch['label']
        image = image.to(self.denoiser.dtype)
        x = (image * 2.0 - 1.0)  # normalize to [-1, 1]

        t, r = self.sample_t_and_r(image.shape[0])
        t_ = rearrange(t, 'b -> b 1 1 1')
        r_ = rearrange(r, 'b -> b 1 1 1')

        noise = torch.randn_like(x, device=self.denoiser.device)
        z = (1 - t_) * x + t_ * noise
        v = noise - x

        call_denoiser = partial(self.denoiser, class_labels=class_labels)

        all_zero = torch.tensor([0.0] * len(x), device=self.denoiser.device)
        all_one = torch.tensor([1.0] * len(x), device=self.denoiser.device)
        u, du_dt = torch.autograd.functional.jvp(
            call_denoiser, 
            (z, t, r), 
            (v, all_one, all_zero),
            create_graph=True,
        )

        target = (v - (t_ - r_) * du_dt).detach()

        loss = 0
        res_dict = {
            'image': batch['image'],
        }
        if 'l2' in self.loss_weights:
            l2_loss = self.loss_weights['l2'] * F.mse_loss(u, target, reduction='mean')
            loss += l2_loss
            self.log(f'{mode}/l2_loss', l2_loss, sync_dist=True)
            res_dict['l2_loss'] = l2_loss.item()
        
        self.log(f'{mode}/loss', loss, sync_dist=True)
        res_dict['loss'] = loss

        return res_dict

    def training_step(self, batch, batch_idx):
        N = self.accumulate_grad_batches
        if self.global_step % 100 == 0 and self.local_rank == 0:
            print(f'INFO: global step {self.global_step}')
        if N == 1:
            res_dict = self.train_internal_step(batch, batch_idx, mode='train')
            if self.use_ema and self.global_step > self.ema_start:
                if (self.local_rank == 0) and (self.global_step % 100 == 0):
                    print(f'INFO: updating EMA model @ step {self.global_step}')
                self.ema_denoiser.update_parameters(self.denoiser)
            return res_dict

        # accumulate gradients with manual optimization (for compatibility with gradient checkpointing)
        opt = self.optimizers()
        res_dict = self.train_internal_step(batch, batch_idx, mode='train')
        loss = res_dict['loss'] / N

        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')

        if (batch_idx + 1) % N == 0:
            opt.step()
            opt.zero_grad()

            if self.use_ema and self.global_step > self.ema_start:
                if (self.local_rank == 0) and (self.global_step % (N * 100) == 0):
                    print(f'INFO: updating EMA model @ step {self.global_step}')
                self.ema_denoiser.update_parameters(self.denoiser)
        
        return res_dict

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        image = batch['image'] # B, H, W for mnist
        height, width = image.shape[2], image.shape[3]
        generated = self.pipe(
            height=height,
            width=width,
            num_samples=image.shape[0],
            class_labels=batch.get('label', None),
            output_type='numpy',
            return_dict=False
        )[0]

        return {
            'generated': generated,
        }

    def on_train_epoch_end(self):
        if self.use_ema and self.global_step > self.ema_start:
            self.denoiser_state_dict = self.denoiser.state_dict()
            ema_state_dict = self.ema_denoiser.module.state_dict()
            self.denoiser.load_state_dict(ema_state_dict)

    def on_train_epoch_start(self):
        if self.use_ema and self.denoiser_state_dict is not None and self.global_step > self.ema_start:
            self.denoiser.load_state_dict(self.denoiser_state_dict)

    def configure_optimizers(self):
        if self.use_8bit_adam:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            self.denoiser.parameters(),
            **self.optim_args
        )
        return optimizer