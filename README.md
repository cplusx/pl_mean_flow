# Mean Flow trianed with pytorch-lightning
The 3rd party [Mean Flow](https://arxiv.org/pdf/2505.13447) paper implementation using `diffusers` (for modules and pipelines) and `pytorch-lightning` (for trainer)

# How to start training
The logging is based on the `wandb`, use `wandb login` and enter your API key.

Use following command 
```python
python train_diffusion.py --config configs/mean_flow_mnist.yaml
```

# Custom training
All the modules are loaded from the config file

# Save as `diffusers` modules

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

The following line gives error
```python
u, du_dt = torch.autograd.functional.jvp(
    call_denoiser, 
    (z, r, t, class_labels), 
    (v, all_zero, all_one, all_zero),
)
```

The solution is to use `functiontools.partial` to wrap the module

```python
call_denoiser = partial(self.denoiser, class_labels=class_labels)
all_zero = torch.tensor([0.0] * len(x), device=self.denoiser.device)
all_one = torch.tensor([1.0] * len(x), device=self.denoiser.device)
u, du_dt = torch.autograd.functional.jvp(
    call_denoiser, 
    (z, r, t), 
    (v, all_zero, all_one),
)
```