# Implementation Guide

1. Fork `samet-akcay/skip-ganomaly`.
2. Add the `src/` overlay package from this bundle.
3. Apply or adapt the patch files under `patches/`.
4. Keep the original Skip-GANomaly code as the baseline model.
5. Add the UNet++ generator, the ensemble wrapper, multiscale dataset support, and UMAP hooks.
6. Train baseline and ++ models separately, then initialize the ensemble either from scratch or from pretrained checkpoints.
