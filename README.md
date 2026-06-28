# Skip-GANomaly++ Overlay for Threat-Object Anomaly Detection

This overlay is designed to sit on top of `samet-akcay/skip-ganomaly`.

It adds:
- multi-scale X-ray data handling,
- a UNet++ generator for `Skip-GANomaly++`,
- an ensemble wrapper with scratch/pretrained initialization modes,
- latent extraction + UMAP plotting,
- Colab-friendly training / inference entrypoints,
- patch files describing how to wire the overlay into the archived base repo.

## Base repo anchors
The upstream repo contains the original Skip-GANomaly implementation in `lib/models/skipganomaly.py`, dataset utilities in `lib/data/datasets.py`, and the standard training / evaluation stack in `train.py`, `options.py`, `lib/models/networks.py`, and related modules.

The paper reports the original Skip-GANomaly, a UNet++-style modified generator, and an ensemble that improves AUC on Compass-XP and SIXray. It also uses augmentation for Compass-XP and patch-based training for SIXray.

## How to use
1. Fork `samet-akcay/skip-ganomaly`.
2. Drop the files in `src/` into your fork (or mirror the structure under `lib/`).
3. Apply the patch files in `patches/` conceptually or adapt them to your exact fork state.
4. Use the Colab notebook to train `skipganomaly`, `skipganomaly_pp`, and `ensemble`.
