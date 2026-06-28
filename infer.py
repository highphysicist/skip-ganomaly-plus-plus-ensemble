from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import XRayFolderDataset
from .data.transforms import build_eval_transforms
from .models.skipganomaly_plus import SkipGanomalyPlus
from .models.ensemble import SkipGanomalyEnsemble
from .utils.checkpoint import load_checkpoint
from .utils.umap_viz import collect_latents, plot_latents


def build_model(kind: str, ckpt: str):
    if kind == 'skipganomaly':
        model = SkipGanomalyPlus()
    else:
        model = SkipGanomalyEnsemble(init_mode='scratch')
    load_checkpoint(ckpt, model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--kind', choices=['skipganomaly', 'ensemble'], default='ensemble')
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--out-dir', default='./eval')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = XRayFolderDataset(args.data_root, args.split, transform=build_eval_transforms(args.image_size), use_labels=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    model = build_model(args.kind, args.checkpoint).to(device)
    model.eval()
    latents, labels, paths = collect_latents(model, loader, device=device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_latents(latents, labels, out / 'umap.png', title=f'{args.kind} latent space')
    pd.DataFrame({'path': paths, 'label': labels}).to_csv(out / 'predictions.csv', index=False)


if __name__ == '__main__':
    main()
