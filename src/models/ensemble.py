from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn

from .skipganomaly_plus import SkipGanomalyPlus
from .score import ScoreWeights, reconstruction_score, latent_score, combine_scores


@dataclass
class EnsembleOutput:
    score: torch.Tensor
    latent_skip: torch.Tensor
    latent_pp: torch.Tensor
    recon_skip: torch.Tensor
    recon_pp: torch.Tensor
    skip_parts: Dict[str, torch.Tensor]
    pp_parts: Dict[str, torch.Tensor]


class SkipGanomalyEnsemble(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, feature_dim=128,
                 init_mode='scratch', skip_ckpt: Optional[str] = None, pp_ckpt: Optional[str] = None,
                 weights: Optional[ScoreWeights] = None):
        super().__init__()
        self.skip = SkipGanomalyPlus(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, feature_dim=feature_dim)
        self.pp = SkipGanomalyPlus(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, feature_dim=feature_dim)
        self.weights = weights or ScoreWeights()
        self.init_mode = init_mode
        self.skip_ckpt = skip_ckpt
        self.pp_ckpt = pp_ckpt
        self._maybe_load()

    def _maybe_load(self):
        if self.init_mode != 'pretrained':
            return
        if self.skip_ckpt:
            state = torch.load(self.skip_ckpt, map_location='cpu')
            self.skip.load_state_dict(state['model'] if 'model' in state else state)
        if self.pp_ckpt:
            state = torch.load(self.pp_ckpt, map_location='cpu')
            self.pp.load_state_dict(state['model'] if 'model' in state else state)

    def forward(self, x):
        skip_out = self.skip(x)
        pp_out = self.pp(x)
        skip_rec = reconstruction_score(x, skip_out.recon)
        skip_lat = latent_score(skip_out.feat_real, skip_out.feat_fake)
        skip_adv = torch.abs(skip_out.pred_real.squeeze(-1) - skip_out.pred_fake.squeeze(-1))
        pp_rec = reconstruction_score(x, pp_out.recon)
        pp_lat = latent_score(pp_out.feat_real, pp_out.feat_fake)
        pp_adv = torch.abs(pp_out.pred_real.squeeze(-1) - pp_out.pred_fake.squeeze(-1))
        skip_fake_on_pp, _ = self.pp.discriminator(skip_out.recon)
        pp_fake_on_skip, _ = self.skip.discriminator(pp_out.recon)
        skip_cross = torch.abs(skip_fake_on_pp.squeeze(-1) - skip_out.pred_fake.squeeze(-1))
        pp_cross = torch.abs(pp_fake_on_skip.squeeze(-1) - pp_out.pred_fake.squeeze(-1))
        skip_parts = {'rec': skip_rec, 'latent': skip_lat, 'adv': skip_adv, 'cross': skip_cross}
        pp_parts = {'rec': pp_rec, 'latent': pp_lat, 'adv': pp_adv, 'cross': pp_cross}
        score_skip = combine_scores(skip_parts, self.weights)
        score_pp = combine_scores(pp_parts, self.weights)
        final = 0.5 * score_skip + 0.5 * score_pp + 0.25 * (skip_cross + pp_cross)
        return EnsembleOutput(final, skip_out.latent, pp_out.latent, skip_out.recon, pp_out.recon, skip_parts, pp_parts)

    @torch.no_grad()
    def score(self, x):
        return self.forward(x)
