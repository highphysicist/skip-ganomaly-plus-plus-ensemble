from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str = './runs'
    image_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 40
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)
    lambda_rec: float = 50.0
    lambda_latent: float = 1.0
    lambda_adv: float = 1.0
    lambda_cross_disc: float = 0.5
    model_name: str = 'skipganomaly'
    ensemble_init: str = 'scratch'
    pretrained_skip: Optional[str] = None
    pretrained_pp: Optional[str] = None
    device: str = 'cuda'
    seed: int = 42
    num_channels: int = 3
    save_umap: bool = True
    use_amp: bool = True

    def resolve(self) -> Path:
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out
