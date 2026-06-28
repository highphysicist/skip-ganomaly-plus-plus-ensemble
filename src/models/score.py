from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ScoreWeights:
    rec: float = 0.5
    latent: float = 0.3
    adv: float = 0.2
    cross: float = 0.2


def reconstruction_score(x, recon):
    return torch.mean(torch.abs(x - recon), dim=(1, 2, 3))


def latent_score(z, z_hat):
    return torch.mean((z - z_hat) ** 2, dim=1)


def combine_scores(parts: Dict[str, torch.Tensor], weights: ScoreWeights):
    return weights.rec * parts['rec'] + weights.latent * parts['latent'] + weights.adv * parts['adv'] + weights.cross * parts['cross']
