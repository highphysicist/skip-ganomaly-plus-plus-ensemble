from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class XRayFolderDataset(Dataset):
    def __init__(self, root: str, split: str, transform: Optional[Callable] = None, use_labels: bool = True):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_labels = use_labels
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f'{split_dir} does not exist')
        self.samples = []
        for cls_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            label = 1 if 'anom' in cls_dir.name.lower() or 'threat' in cls_dir.name.lower() else 0
            for p in cls_dir.rglob('*'):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, label))
        if not self.samples:
            raise RuntimeError(f'No image files found under {split_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if not self.use_labels:
            label = 0
        return img, label, str(path)
