# Modified from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/dit/pipeline_dit.py#L31

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from diffusers.models import AutoencoderKL, DiTTransformer2DModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class MeanFlowPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "denoiser->vae"

    def __init__(
        self,
        denoiser: DiTTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(denoiser=denoiser)

    @torch.no_grad()
    def __call__(
        self,
        images: List[Union[np.array, torch.Tensor]],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:

        if isinstance(images[0], np.ndarray):
            images = torch.from_numpy(images).to(self._execution_device)

        if images.max() > 1:
            images = images / 255.0

        batch_size = len(images)
        # image_cond = self.encode_image_to_latent(images) / self.vae.config.scaling_factor
        image_cond = self.encode_condition_image(images)
        latent_size_h, latent_size_w = image_cond.shape[-2:]
        latent_channels = self.denoiser.config.out_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size_h, latent_size_w),
            generator=generator,
            device=self._execution_device,
            dtype=self.denoiser.dtype,
        )

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                latents_concat = torch.cat([latents, image_cond], dim=1)
                latent_concat_uncond = torch.cat([latents, torch.zeros_like(image_cond)], dim=1)
                latent_model_input = torch.cat([latents_concat, latent_concat_uncond], dim=0)
            else:
                latents_concat = torch.cat([latents, image_cond], dim=1)
                latent_model_input = latents_concat
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # remove it, both DDIM and Flow Matching do not use it

            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output
            noise_pred, _ = self.denoiser(
                latent_model_input, timestep=timesteps, return_dict=False
            ) # _ is zs, for the REPA

            # perform guidance
            if guidance_scale > 1:
                cond_eps, uncond_eps = torch.split(noise_pred, len(noise_pred) // 2, dim=0)
                noise_pred = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

            # compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        samples = self.decode_latent_to_image(latents)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
