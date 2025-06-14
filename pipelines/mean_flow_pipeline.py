# Modified from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/dit/pipeline_dit.py#L31

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.mean_flow_model import DualTimestepDiTTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class MeanFlowPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "denoiser"

    def __init__(
        self,
        denoiser: DualTimestepDiTTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(denoiser=denoiser)

    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        num_samples: int = 1,
        class_labels: Optional[torch.LongTensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:

        batch_size = num_samples
        e = randn_tensor(
            shape=(batch_size, 1, height, width),
            generator=generator,
            device=self._execution_device,
            dtype=self.denoiser.dtype,
        )

        if class_labels is None:
            class_labels = torch.randint(
                low=0,
                high=10,
                size=(batch_size,),
                device=self._execution_device,
                dtype=torch.long,
            )

        # set step values
        noise_pred = self.denoiser(
            e, 
            t=torch.tensor([1] * batch_size, device=self._execution_device, dtype=torch.long),
            r=torch.tensor([0] * batch_size, device=self._execution_device, dtype=torch.long),
            class_labels=class_labels,
        )

        samples = e - noise_pred # range -1, 1
        samples = (samples + 1.0) / 2.0  # normalize to [0, 1]

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
