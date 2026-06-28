from __future__ import annotations

import random
from typing import Sequence

from PIL import Image
from torchvision import transforms as T


class MultiScaleResize:
    def __init__(self, sizes: Sequence[int]):
        self.sizes = list(sizes)
        if not self.sizes:
            raise ValueError('sizes must not be empty')

    def __call__(self, img: Image.Image) -> Image.Image:
        size = random.choice(self.sizes)
        return img.resize((size, size), Image.BICUBIC)


def build_train_transforms(image_sizes: Sequence[int], augment: bool = True):
    ops = [MultiScaleResize(image_sizes)]
    if augment:
        ops.extend([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        ])
    ops.extend([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return T.Compose(ops)


def build_eval_transforms(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
