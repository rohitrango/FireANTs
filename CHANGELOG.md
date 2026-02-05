# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Fixed

- **Data type mismatch**: Fixed data type mismatch in several registration classes (abstract.py, distributedgreedy.py, greedy.py, syn.py). Reference: #77

### Added

- **Masked losses**: Support for region-of-interest (ROI) driven registration using masked variants of CC and MSE.
  - Concatenate a mask as the last channel to fixed and moving images; the loss is computed only where the mask is non-zero.
  - Use `loss_type="masked_cc"` or `loss_type="masked_mse"` to enable masked mode (no extra options required).
  - Helper `generate_image_mask_allones(image)` for images that have no mask (use with `apply_mask_to_image` so both images have a mask channel).
  - Gaussian smoothing in multi-resolution registration is applied only to image channels, not the mask channel, when masked mode is active.
  - See [How To: Masked losses](docs/docs/howto/masked-losses.md) in the documentation.
