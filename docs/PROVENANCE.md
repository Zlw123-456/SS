# Provenance and Scope

## Included additions

The package isolates the authors' band-allocation rules and shared band-importance gate from the speech research implementation, where these additions were embedded alongside a TIGER-based model. Public files have standalone names and English documentation. No complete source file from that model is included.

| Research name | Standalone name | Treatment |
| --- | --- | --- |
| `build_exp_band` | `build_exp_band` | Preserved arithmetic and integer correction order; added explicit input validation |
| `build_hybrid_band` | `build_hybrid_band` | Preserved low-bin/high-band construction and defaults |
| `BandImportanceGate` | `BandRecalibration` | Preserved the current research gate's pooling, SiLU MLP, scaled sigmoid, initialization, multiplication, and state dictionary names |
| No standalone helper | `band_slices` | Added index-interval helper; does not implement an encoder or backbone |

Input validation now rejects noninteger counts, invalid shapes, nonfinite growth parameters, and numerically overflowing settings. The single-high-band case is explicit. These changes clarify invalid-input behavior and do not intentionally change valid experimental configurations.

The BR description and standalone implementation have been synchronized with the revised research implementation used for retraining. BA arithmetic is unchanged. The public names remain BA and BR. Earlier BR snapshots with ReLU and an unscaled sigmoid are not mathematically equivalent to the current implementation.

The locally inspected research archive has SHA-256 `4baa91aaa45e65a17e2552e10a760a5737ec4f8b0e711aed9eacbce1f1e21a8a`. This digest identifies the source used for the local comparison, not a public download or a model checkpoint. The original archive is not redistributed.

## Explicitly excluded

- TIGER backbone/model classes and its frequency-frame attention or multi-scale components.
- The host's band encoder, reconstruction layers, STFT/iSTFT wrapper, or mask heads.
- Historical Top-K selection, head masking, and other backbone ablations.
- Loss functions, data loaders, training scripts, evaluation pipelines, or external datasets.
- Music-separation code or configurations.
- Full-model or standalone trained weights, logs, credentials, machine paths, private hostnames, and unpublished paper files.

TIGER is acknowledged as an external experimental backbone, not as code redistributed by this package. Its authors' work and applicable terms should be acknowledged separately when acquiring or using their implementation.

This scope statement is not a legal determination of ownership. Authors and the institution should review the selected code and license before publication.
