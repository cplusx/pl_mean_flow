'''
Example: python pl_weights_to_diffusers_pipeline.py \
    --pl_ckpt experiments/mean_flow_mnist/last.ckpt/checkpoint/mp_rank_00_model_states.pt \
    --config configs/mean_flow_mnist.yaml \
    --save_name mean_flow_mnist

Usage:
from pipelines.mean_flow_pipeline import MeanFlowPipeline
pipe = MeanFlowPipeline.from_pretrained('mean_flow_mnist')
generated = pipe(
    height=28,
    width=28,
    num_samples=4,
    output_type='np',
)
print(generated.images.shape) # should be (4, 28, 28, 1)
'''
import argparse
import torch
from misc_utils.train_utils import unit_test_create_diffusion_model

parser = argparse.ArgumentParser()
parser.add_argument('--pl_ckpt', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save_name', type=str, required=True)

args = parser.parse_args()

pl_trainer = unit_test_create_diffusion_model(args.config)

weights = torch.load(args.pl_ckpt, map_location='cpu', weights_only=False)['module']

ema_ckpt = {k.replace('ema_denoiser.module.', ''): v for k, v in weights.items() if 'ema_denoiser.module.' in k}
pl_trainer.pipe.denoiser.load_state_dict(ema_ckpt)
pl_trainer.pipe.save_pretrained(args.save_name)