from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def conv_norm_relu(in_ch, out_ch, kernel=3, stride=1, padding=1, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(conv_norm_relu(in_ch, out_ch), conv_norm_relu(out_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, feature_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(base_ch * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(base_ch * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(base_ch * 8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature = nn.Linear(base_ch * 8, feature_dim)
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        feat_map = self.backbone(x)
        pooled = self.pool(feat_map).flatten(1)
        feat = self.feature(pooled)
        logit = self.head(torch.relu(feat))
        return torch.sigmoid(logit), feat
