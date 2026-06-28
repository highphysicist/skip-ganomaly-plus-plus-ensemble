from __future__ import annotations

from typing import Any, Dict

import torch


def save_checkpoint(path: str, model, optimizer=None, epoch: int = 0, extra: Dict[str, Any] | None = None):
    payload = {'epoch': epoch, 'model': model.state_dict()}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None, map_location='cpu'):
    payload = torch.load(path, map_location=map_location)
    state = payload['model'] if 'model' in payload else payload
    model.load_state_dict(state)
    if optimizer is not None and 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])
    return payload
