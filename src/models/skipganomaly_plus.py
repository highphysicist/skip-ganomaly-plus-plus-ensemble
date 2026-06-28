from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .unetpp import UNetPlusPlusGenerator
from .blocks import PatchDiscriminator
from .score import reconstruction_score, latent_score


@dataclass
class SingleModelOutput:
    recon: torch.Tensor
    latent: torch.Tensor
    pred_real: torch.Tensor
    pred_fake: torch.Tensor
    feat_real: torch.Tensor
    feat_fake: torch.Tensor


class SkipGanomalyPlus(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, feature_dim=128):
        super().__init__()
        self.generator = UNetPlusPlusGenerator(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)
        self.discriminator = PatchDiscriminator(in_ch=out_ch, base_ch=base_ch, feature_dim=feature_dim)

    def forward(self, x):
        recon, latent = self.generator(x, return_latent=True)
        pred_real, feat_real = self.discriminator(x)
        pred_fake, feat_fake = self.discriminator(recon)
        return SingleModelOutput(recon, latent, pred_real, pred_fake, feat_real, feat_fake)

    @torch.no_grad()
    def score(self, x):
        out = self.forward(x)
        rec = reconstruction_score(x, out.recon)
        lat = latent_score(out.feat_real, out.feat_fake)
        adv = torch.abs(out.pred_real.squeeze(-1) - out.pred_fake.squeeze(-1))
        return rec + lat + adv, out.latent
