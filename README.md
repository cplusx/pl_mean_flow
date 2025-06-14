# Mean Flow trianed with pytorch-lightning
The 3rd party [Mean Flow](https://arxiv.org/pdf/2505.13447) paper implementation using `diffusers` (for modules and pipelines) and `pytorch-lightning` (for trainer)

## Table of Contents

- [Training demo for MNIST](#training-demo-for-mnist)
- [Custom training](#custom-training)
- [Save as `diffusers` modules](#save-as-diffusers-modules)
- [Solutions for compatiblity with diffusers and pytorch](#compatiblity-with-diffusers-and-pytorch)

# Training demo for MNIST
The logging is based on the `wandb`, use `wandb login` and enter your API key.

Use following command 
```python
python train_diffusion.py --config configs/mean_flow_mnist.yaml
```

# Custom training
All the modules are loaded from the config file. Edit the config YAML file to customize these options for your experiment. The following are configurations that you may be interested to customize.

| Option         | Description                          | Example
|----------------|---------------------------------------------|------
| `trainer_args`        | Specifies the device, epochs, etc... for pytorch-lightning | See config file
| `diffusion_trainer`   | The model trainer | `pl_trainer.mean_flow_trainer.MeanFlowTrainer`
| `pipe`                | The `diffusers` style pipeline that can be pushed to remote and shared with others    | `pipelines.mean_flow_pipeline.MeanFlowPipeline`
| `denoiser`            | The main training model   | `models.mean_flow_model.DualTimestepDiTTransformer2DModel`
| `data`                | The data. You may need to change accordingly in the `trainer`'s `training_step` to process your data correctly. | see config file

### Model Trainer (`diffusion_trainer`)
The following functionalities are supported in the model trainer
| Func | Example
|------|-------
| gradient accumulation | `accumulate_grad_batches: 8`
| moving average |     `use_ema: True ema_decay: 0.99 ema_start: 500`
| gradient_checkpoint | `gradient_checkpointing: True` (if your model supports)
| 8 bits training | `use_8bit_adam: True` (need `bitsandbytes` installed)


### Denoiser Model (`denoiser`)
This demo uses the model script of `diffusers.DiTTransformer2DModel` but supports the two timestep inputs by replacing the time-conditioned attention layers. Refer to foler [`models`](models) for details.

# Save as `diffusers` pipeline/modules
The trained weights will be saved in `experiments/mean_flow_mnist/last.ckpt/checkpoint/mp_rank_00_model_states.pt`
You can use `pl_weights_to_diffusers_pipeline.py` to convert it to `diffusers` pipeline and share it online.
```python
# Example to save MNIST mean flow model
python pl_weights_to_diffusers_pipeline.py \
    --pl_ckpt experiments/mean_flow_mnist/last.ckpt/checkpoint/mp_rank_00_model_states.pt \
    --config configs/mean_flow_mnist.yaml \
    --save_name mean_flow_mnist
```

To use the model, just use `from_pretrained()` method in the diffusers.
```python
# Usage:
from pipelines.mean_flow_pipeline import MeanFlowPipeline
pipe = MeanFlowPipeline.from_pretrained('mean_flow_mnist')
generated = pipe(
    height=28,
    width=28,
    num_samples=4,
    output_type='np',
)
print(generated.images.shape) # should be (4, 28, 28, 1)
```

# Compatiblity with diffusers and pytorch
## How to solve flash attention problem
The `torch.autograd.functional.jvp` does not support flash attention, the solution is to set the `Attention` layers in the `diffusers` to use legacy processor `AttnProcessor()`
```python
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor

# in the __init__ of the trainer
for name, m in self.denoiser.named_modules():
    if isinstance(m, Attention):
        m.set_processor(AttnProcessor()) # compatible with jvp
```


## How to solve class_labels input problem
The `torch.autograd.functional.jvp` cannot accept an input that has no gradient, such as class_labels.

The following line gives error because `class_labels` is a long type.
```python
u, du_dt = torch.autograd.functional.jvp(
    call_denoiser, 
    (z, t, r, class_labels), 
    (v, all_one, all_zero, all_zero),
)
```

The solution is to use `functiontools.partial` to wrap the module

```python
call_denoiser = partial(self.denoiser, class_labels=class_labels)
all_zero = torch.tensor([0.0] * len(x), device=self.denoiser.device)
all_one = torch.tensor([1.0] * len(x), device=self.denoiser.device)
u, du_dt = torch.autograd.functional.jvp(
    call_denoiser, 
    (z, t, r), 
    (v, all_one, all_zero),
)
```