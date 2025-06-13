import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MeanFlowTrainer(pl.LightningModule):
    def __init__(
        self,
        pipe, 
        cond_dropout: float=0.1,
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pipe = pipe
        self.vae: AutoencoderKL = pipe.vae
        self.denoiser: DiTTransformer2DModelREPA = pipe.denoiser
        self.cond_encoder: nn.Module = pipe.cond_encoder
        self.cond_dropout = cond_dropout
        self.guidance_scale = guidance_scale
        self.optim_args = optim_args
        self.loss_weights = loss_weights
        self.use_8bit_adam = use_8bit_adam
        self.accumulate_grad_batches = accumulate_grad_batches
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

    def train_internal_step(self, batch, batch_idx, mode='train'):
        image = batch['image'] # should be in [0, 1], (B, C, H, W)
        edge = batch['edge']
        image = image.to(self.vae.dtype)
        edge = edge.to(self.vae.dtype)
        image_cond = self.encode_condition_image(image)
        edge_latent = self.encode_image_to_latent(edge)

        image_cond = image_cond.to(self.denoiser.dtype)
        edge_latent = edge_latent.to(self.denoiser.dtype)
        timesteps = self.get_timesteps(len(image), self.num_timesteps).to(edge_latent.device, edge_latent.dtype)
        noisy_latent, target = self.add_noise_and_get_target(edge_latent, timesteps)

        if torch.rand(1).item() < self.cond_dropout:
            image_cond = torch.zeros_like(image_cond)

        model_input = torch.cat([noisy_latent, image_cond], dim=1)
        model_pred, zs = self.denoiser(model_input, timesteps, return_dict=False)

        edge_latent_pred = self.get_x0_from_xt(noisy_latent, timesteps, model_pred)
        edge_pred = self.decode_latent_to_image(edge_latent_pred)

        loss = 0
        res_dict = {
            'image': image,
            'edge': edge,
            'edge_pred': edge_pred,
        }
        if 'l1' in self.loss_weights:
            l1_loss = self.l1_loss(model_pred, target) * self.loss_weights['l1']
            loss += l1_loss
            self.log(f'{mode}/l1_loss', l1_loss, sync_dist=True)
            res_dict['l1_loss'] = l1_loss.item()

        if 'l2' in self.loss_weights:
            l2_loss = self.l2_loss(model_pred, target) * self.loss_weights['l2']
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
        generated = self.pipe(
            # TODO, specify kwargs
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