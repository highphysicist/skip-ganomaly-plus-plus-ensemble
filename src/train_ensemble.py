from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from .utils.config import TrainConfig
from .data.dataset import XRayFolderDataset
from .data.transforms import build_train_transforms
from .models.ensemble import SkipGanomalyEnsemble
from .models.train_runner import train_ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out-dir', default='./runs/ensemble')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--image-sizes', nargs='+', type=int, default=[32, 64, 128, 256, 512])
    ap.add_argument('--ensemble-init', choices=['scratch', 'pretrained'], default='scratch')
    ap.add_argument('--skip-ckpt', default=None)
    ap.add_argument('--pp-ckpt', default=None)
    args = ap.parse_args()
    cfg = TrainConfig(data_root=args.data_root, out_dir=args.out_dir, epochs=args.epochs, batch_size=args.batch_size, device=args.device, image_sizes=args.image_sizes, ensemble_init=args.ensemble_init, pretrained_skip=args.skip_ckpt, pretrained_pp=args.pp_ckpt)
    train_ds = XRayFolderDataset(args.data_root, 'train', transform=build_train_transforms(cfg.image_sizes, augment=True), use_labels=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    model = SkipGanomalyEnsemble(init_mode=cfg.ensemble_init, skip_ckpt=cfg.pretrained_skip, pp_ckpt=cfg.pretrained_pp)
    train_ensemble(model, train_loader, cfg, out_dir=cfg.out_dir)


if __name__ == '__main__':
    main()
