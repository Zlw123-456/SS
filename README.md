# ABAR: Speech Separation Modules

Standalone **Band Allocation (BA)** and **Band Recalibration (BR)** components for speech separation.

[中文说明](README_zh-CN.md) | [Methods and API](docs/METHODS.md) | [Integration](docs/INTEGRATION.md) | [Release checklist](docs/RELEASE_CHECKLIST.md)

> **Scope:** this repository contains only the added allocation rules and feature-recalibration module. It does **not** contain the TIGER model, its encoder/separator/mask heads, a complete training pipeline, pretrained weights, datasets, or music-separation code. This is a component release, not an end-to-end reproduction repository.
>
> **License status:** a license has not yet been selected. This package is prepared for author/supervisor review before public release; no open-source license grant is included. See [Licensing](#licensing).

## Components

| Component | File | Purpose |
| --- | --- | --- |
| Exponential allocation | `abar_speech/allocation.py` | Integer bandwidths with exact coverage and the experimental correction order |
| Hybrid allocation (BA) | `abar_speech/allocation.py` | Single-bin low-frequency bands followed by exponential high-frequency bands |
| Recalibration (BR) | `abar_speech/recalibration.py` | Shared temporal-pooling MLP producing one continuous weight per band |
| Speech configuration | `configs/speech_ba_br.json` | 16-kHz, 321-bin, 67-band module settings |
| Examples | `examples/` | Allocation inspection and synthetic feature forward/backward checks |
| Tests | `tests/` | Coverage, rounding, shapes, broadcasting, gradients, and state dictionaries |

BA was called **HybridBand**, and BR was called **BandGate**, in the development implementation. This release uses the paper's BA/BR terminology. The exponential rule is a building block of BA, not a third proposed module. There is no hard selection or Top-K operation in this package.

## Installation

Python 3.10 or newer is required. From this repository's root:

```bash
# BA only; does not install or import PyTorch.
python -m pip install -e .

# BA and BR. For a particular CUDA/CPU build, install PyTorch first.
python -m pip install -e ".[br]"
```

Select the PyTorch build appropriate for your environment using its [official installation guide](https://pytorch.org/get-started/locally/). This repository does not install TIGER or any other separation backbone.

## Quick Start

### Band allocation

```python
from abar_speech import band_slices, build_hybrid_band

widths = build_hybrid_band(total_bins=321, low_bins=60, total_bands=67, alpha=2.0)
assert widths == [1] * 60 + [11, 15, 21, 30, 43, 59, 82]
assert len(widths) == 67 and sum(widths) == 321
regions = band_slices(widths)
# For a spectrum X shaped [batch, frequency, time]: X[:, regions[k], :]
```

The regions are contiguous half-open bin intervals. BA changes the grouping of STFT bins, not the STFT's physical frequency resolution. Boundaries are fixed before training and remain unchanged during inference.

### Band recalibration

```python
import torch
from abar_speech.recalibration import BandRecalibration

br = BandRecalibration(feature_dim=128, hidden_dim=64)
z = torch.randn(2, 67, 128, 20)  # [batch, bands, features, time]
z_rec, weights = br(z, return_gate=True)
assert z_rec.shape == z.shape
assert weights.shape == (2, 67)
```

BR first averages each band's features over time, then applies a shared two-layer MLP with a **SiLU** activation. A scaled sigmoid converts the resulting logit into a scalar weight:

```text
p_k = mean_t(z_k[:, t])
u_k = W2 * SiLU(W1 * p_k + b1) + b2
g_k = 2 * sigmoid(u_k)
z_rec_k = g_k * z_k
```

The mathematical gate range is **(0, 2)**: values below one attenuate features, while values above one amplify them. BR contains **8,321 trainable parameters** for this configuration. All bands are retained. Gate weights describe scaling of learned features, not calibrated physical frequency importance.

Both linear-layer biases are initialized to zero. The first weight matrix keeps PyTorch's default initialization; the final weight matrix is sampled from a zero-mean normal distribution with standard deviation `1e-3`. Small initial logits therefore yield weights close to one. This is **near-identity initialization**, not a residual connection or an exact identity guarantee for arbitrary inputs.

## Examples and Tests

```bash
python examples/inspect_bands.py
python examples/demo_recalibration.py
python -m unittest discover -s tests -v
```

The BR example and full test suite require the `br` dependency. BA-only tests can be run with:

```bash
python -m unittest discover -s tests -p test_allocation.py -v
```

The feature example uses random tensors and does not train or evaluate a speech-separation model. See [validation notes](docs/VALIDATION.md) for the checks performed on this package.

## Integration and Reproducibility

Use BA's widths consistently in your own band extraction, encoding, and reconstruction interfaces. Apply BR to the stacked encoded features **before** the separator. Keep the original complex mixture sub-band spectra available for final mask application; BR does not replace those spectra. See [integration details](docs/INTEGRATION.md).

The configuration is a **module configuration**, not a drop-in TIGER training YAML. The paper's speech settings include a 640-sample STFT window/FFT and a 160-sample hop, with a 128-dimensional encoded representation. Training, dataset preparation, evaluation metrics, and backbone dependencies must be supplied separately. Running these examples cannot reproduce the paper's SDRi/SI-SNRi results.

The experiments were retrained after updating BR. The current definition uses SiLU, `2 * sigmoid`, and the initialization described above; BA is unchanged. Earlier checkpoints and results must not be treated as results of this implementation merely because the public module name remains BR. This package does not include updated numerical separation results or full-model checkpoints. See [experimental settings and result provenance](docs/EXPERIMENTS.md).

**Checkpoint compatibility:** the earlier gate used ReLU and an unscaled sigmoid. Its trainable parameter names and shapes match the current gate, so even a successful strict state-dictionary load does not establish equivalent behavior. Use the implementation and checkpoint that belong to the same experiment.

## Attribution

The speech experiments use [TIGER](https://github.com/JusperLee/TIGER) as an external backbone. Refer to the [TIGER paper](https://arxiv.org/abs/2410.01469) and its repository for the upstream implementation and applicable terms. This package does not redistribute that implementation. See [provenance and scope](docs/PROVENANCE.md).

The project repository is [Zlw123-456/SS](https://github.com/Zlw123-456/SS). `CITATION.cff` describes this software package. A paper DOI or proceedings entry should only be added once confirmed; none is invented here.

## Licensing

No license has been selected or granted in this review package. The authors and their institution should approve the copyright attribution and license before public release. An upstream project's license does not automatically license the additions here. Do not describe this package as licensed open-source software until that decision is recorded in a `LICENSE` file.
