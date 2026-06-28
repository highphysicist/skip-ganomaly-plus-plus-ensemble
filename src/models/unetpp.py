from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import DoubleConv, Down, OutConv


class UNetPlusPlusGenerator(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 64):
        super().__init__()
        filters = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.x0_0 = DoubleConv(in_ch, filters[0])
        self.x1_0 = Down(filters[0], filters[1])
        self.x2_0 = Down(filters[1], filters[2])
        self.x3_0 = Down(filters[2], filters[3])
        self.x4_0 = Down(filters[3], filters[4])

        self.x0_1 = DoubleConv(filters[0] + filters[1], filters[0])
        self.x1_1 = DoubleConv(filters[1] + filters[2], filters[1])
        self.x2_1 = DoubleConv(filters[2] + filters[3], filters[2])
        self.x3_1 = DoubleConv(filters[3] + filters[4], filters[3])

        self.x0_2 = DoubleConv(filters[0] * 2 + filters[1], filters[0])
        self.x1_2 = DoubleConv(filters[1] * 2 + filters[2], filters[1])
        self.x2_2 = DoubleConv(filters[2] * 2 + filters[3], filters[2])

        self.x0_3 = DoubleConv(filters[0] * 3 + filters[1], filters[0])
        self.x1_3 = DoubleConv(filters[1] * 3 + filters[2], filters[1])

        self.x0_4 = DoubleConv(filters[0] * 4 + filters[1], filters[0])
        self.out_conv = OutConv(filters[0], out_ch)
        self.latent_proj = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(filters[4], 256), nn.ReLU(inplace=True))

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=True)

    def forward(self, x, return_latent: bool = True):
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(x0_0)
        x2_0 = self.x2_0(x1_0)
        x3_0 = self.x3_0(x2_0)
        x4_0 = self.x4_0(x3_0)

        x0_1 = self.x0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))
        x1_1 = self.x1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x2_1 = self.x2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x3_1 = self.x3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))

        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x2_2 = self.x2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))

        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        x1_3 = self.x1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))

        x0_4 = self.x0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))
        recon = torch.tanh(self.out_conv(x0_4))
        latent = self.latent_proj(x4_0)
        return (recon, latent) if return_latent else recon
