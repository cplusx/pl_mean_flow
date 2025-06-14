import numpy as np
from torchvision.datasets import MNIST

class MNISTDataset(MNIST):

    def __getitem__(self, index):
        pil_image, target = super().__getitem__(index)
        return {
            'image': np.array(pil_image, dtype=np.float32)[None] / 255.0,  # Normalize to [0, 1], 1, H, W
            'label': target
        }