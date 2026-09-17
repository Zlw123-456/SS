# Integration into an External Speech Model

This package deliberately does not ship an encoder, separator, mask head, decoder, loss implementation, or training loop. The steps below describe interfaces; acquire and maintain the host model separately.

## 1. Construct the partition before building the host model

For a 640-point real-input STFT, `F = 640 // 2 + 1 = 321`. Build the widths once:

```python
from abar_speech import band_slices, build_hybrid_band

widths = build_hybrid_band(321, low_bins=60, total_bands=67, alpha=2.0)
regions = band_slices(widths)
```

The host must use these same widths when extracting mixture sub-bands, constructing band-specific encoders, constructing mask outputs, and reassembling the spectrum. Changing only the slicing at inference can mismatch trained encoder/mask dimensions. A checkpoint trained with different boundaries is not automatically compatible, even when `F` and `K` match.

## 2. Obtain encoded features from the host

The host converts each complex sub-band into real-valued features with a common dimension `D`. Stack them in increasing frequency order to produce `[B, K, D, T]`. This package does not prescribe or redistribute the host's encoding layers.

## 3. Apply BR before the separator

Register `BandRecalibration` as a submodule of your own model during initialization, before constructing the optimizer:

```python
from abar_speech.recalibration import BandRecalibration

# In your host model's __init__:
self.band_recalibration = BandRecalibration(feature_dim=128, hidden_dim=64)

# In your host model's forward, after obtaining encoded_bands [B, K, D, T]:
recalibrated_bands = self.band_recalibration(encoded_bands)
# Pass recalibrated_bands to the host's existing separator.
```

These lines are integration guidance, not a complete runnable model. If your host uses `[B, D, K, T]`, permute the band and feature axes before BR and restore its expected layout afterward. Do not construct a fresh gate inside `forward`, and ensure its parameters are included in the optimizer.

## 4. Keep reconstruction separate from recalibration

Retain the original complex mixture sub-band spectra. The host predicts source-specific complex masks from the separated features, applies masks to the corresponding original spectra, concatenates the bands, and performs inverse STFT. BR scales encoded features; it does not directly gate the original waveform or substitute for a complex mask.

## 5. Use controlled comparisons

For a BA-only experiment, use the new boundaries without BR. For BA+BR, use the same BA boundaries and enable BR. Keep the backbone configuration, data split, optimizer protocol, and evaluation procedure controlled. A fixed number of bands does not guarantee an identical total parameter count for every possible host architecture; verify this for your chosen host.

## Loading a previously trained gate

Only load checkpoints from a trusted source. This package does not include weights or a checkpoint-loading utility. The standalone gate retains the original `scorer` parameter names. If a full-model state dictionary prefixes these keys with `band_gate.`, extract only that gate's state dictionary in your own code and load it with `strict=True`. Confirm the feature dimension, hidden dimension, band definition, and associated encoder before claiming the same trained behavior.
