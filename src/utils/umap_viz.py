from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def collect_latents(model, loader, device='cuda'):
    model.eval()
    latents = []
    labels = []
    paths = []
    with torch.no_grad():
        for x, y, p in loader:
            x = x.to(device)
            out = model.score(x)
            if hasattr(out, 'latent_skip'):
                z = torch.cat([out.latent_skip, out.latent_pp], dim=1)
            else:
                z = out[1]
            latents.append(z.detach().cpu())
            labels.append(torch.as_tensor(y))
            paths.extend(list(p))
    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return latents, labels, paths


def _reduce(latents, random_state=42):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(latents), 'UMAP'
    except Exception:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(latents), 'PCA'


def plot_latents(latents, labels, out_path: str, title: str = 'Latent Space'):
    emb, method = _reduce(latents)
    plt.figure(figsize=(8, 6))
    for lab in sorted(set(labels.tolist())):
        idx = labels == lab
        plt.scatter(emb[idx, 0], emb[idx, 1], s=10, alpha=0.8, label=f'class {lab}')
    plt.title(f'{title} ({method})')
    plt.legend()
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()
    return out
