from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..utils.checkpoint import save_checkpoint


def train_single_model(model, train_loader, val_loader, cfg, out_dir: str):
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device != 'cpu' else 'cpu')
    model = model.to(device)
    model.train()
    opt_d = optim.Adam(model.discriminator.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_g = optim.Adam(model.generator.parameters(), lr=cfg.lr, betas=cfg.betas)
    scaler = GradScaler(enabled=cfg.use_amp and device.type == 'cuda')
    bce = nn.BCELoss()
    l1 = nn.L1Loss()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for epoch in range(cfg.epochs):
        running = 0.0
        for x, y, _ in tqdm(train_loader, desc=f'epoch {epoch+1}/{cfg.epochs}'):
            x = x.to(device)
            real = torch.ones((x.size(0), 1), device=device)
            fake = torch.zeros((x.size(0), 1), device=device)
            with autocast(enabled=scaler.is_enabled()):
                recon, latent = model.generator(x, return_latent=True)
                pred_real, feat_real = model.discriminator(x)
                pred_fake, feat_fake = model.discriminator(recon)
                loss_g = cfg.lambda_adv * bce(pred_fake, real) + cfg.lambda_rec * l1(recon, x) + cfg.lambda_latent * torch.mean((feat_fake - feat_real) ** 2)
                loss_d = bce(pred_real, real) + bce(pred_fake.detach(), fake)
                loss = loss_g + loss_d
            opt_d.zero_grad(set_to_none=True)
            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt_d)
            scaler.step(opt_g)
            scaler.update()
            running += float(loss.item())
        avg = running / max(1, len(train_loader))
        save_checkpoint(out_dir / f'epoch_{epoch+1}.pt', model, extra={'epoch': epoch+1, 'avg_loss': avg})
        if avg < best:
            best = avg
            save_checkpoint(out_dir / 'best.pt', model, extra={'epoch': epoch+1, 'avg_loss': avg})


def train_ensemble(model, train_loader, cfg, out_dir: str):
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device != 'cpu' else 'cpu')
    model = model.to(device)
    model.train()
    params = list(model.skip.parameters()) + list(model.pp.parameters())
    opt = optim.Adam(params, lr=cfg.lr, betas=cfg.betas)
    scaler = GradScaler(enabled=cfg.use_amp and device.type == 'cuda')
    bce = nn.BCELoss()
    l1 = nn.L1Loss()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for epoch in range(cfg.epochs):
        running = 0.0
        for x, y, _ in tqdm(train_loader, desc=f'ensemble epoch {epoch+1}/{cfg.epochs}'):
            x = x.to(device)
            real = torch.ones((x.size(0), 1), device=device)
            with autocast(enabled=scaler.is_enabled()):
                out = model.forward(x)
                skip_out = model.skip(x)
                pp_out = model.pp(x)
                loss_branch = (
                    cfg.lambda_rec * (l1(skip_out.recon, x) + l1(pp_out.recon, x))
                    + cfg.lambda_latent * (torch.mean((skip_out.feat_fake - skip_out.feat_real) ** 2) + torch.mean((pp_out.feat_fake - pp_out.feat_real) ** 2))
                    + cfg.lambda_adv * (bce(skip_out.pred_fake, real) + bce(pp_out.pred_fake, real))
                )
                loss_cross = cfg.lambda_cross_disc * torch.mean(out.score)
                loss = loss_branch + loss_cross
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
        avg = running / max(1, len(train_loader))
        save_checkpoint(out_dir / f'epoch_{epoch+1}.pt', model, extra={'epoch': epoch+1, 'avg_loss': avg})
        if avg < best:
            best = avg
            save_checkpoint(out_dir / 'best.pt', model, extra={'epoch': epoch+1, 'avg_loss': avg})
