from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from .utils.config import TrainConfig
from .data.dataset import XRayFolderDataset
from .data.transforms import build_train_transforms
from .models.skipganomaly_plus import SkipGanomalyPlus
from .models.train_runner import train_single_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out-dir', default='./runs/single')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--image-sizes', nargs='+', type=int, default=[32, 64, 128, 256, 512])
    args = ap.parse_args()
    cfg = TrainConfig(data_root=args.data_root, out_dir=args.out_dir, epochs=args.epochs, batch_size=args.batch_size, device=args.device, image_sizes=args.image_sizes)
    train_ds = XRayFolderDataset(args.data_root, 'train', transform=build_train_transforms(cfg.image_sizes, augment=True), use_labels=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    model = SkipGanomalyPlus()
    train_single_model(model, train_loader, None, cfg, out_dir=cfg.out_dir)


if __name__ == '__main__':
    main()
