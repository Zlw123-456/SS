# Provenance and Scope

## Included additions

The package isolates the authors' band-allocation rules and shared band-importance gate from the speech research implementation, where these additions were embedded alongside a TIGER-based model. Public files have standalone names and English documentation. No complete source file from that model is included.

| Research name | Standalone name | Treatment |
| --- | --- | --- |
| `build_exp_band` | `build_exp_band` | Preserved arithmetic and integer correction order; added explicit input validation |
| `build_hybrid_band` | `build_hybrid_band` | Preserved low-bin/high-band construction and defaults |
| `BandImportanceGate` | `BandRecalibration` | Preserved pooling, MLP, multiplication, initialization, and gate state dictionary names |
| No standalone helper | `band_slices` | Added index-interval helper; does not implement an encoder or backbone |

Input validation now rejects noninteger counts, invalid shapes, nonfinite growth parameters, and numerically overflowing settings. The single-high-band case is explicit. These changes clarify invalid-input behavior and do not intentionally change valid experimental configurations.

## Explicitly excluded

- TIGER backbone/model classes and its frequency-frame attention or multi-scale components.
- The host's band encoder, reconstruction layers, STFT/iSTFT wrapper, or mask heads.
- Historical Top-K selection, head masking, and other backbone ablations.
- Loss functions, data loaders, training scripts, evaluation pipelines, or external datasets.
- Music-separation code or configurations.
- Full-model or standalone trained weights, logs, credentials, machine paths, private hostnames, and unpublished paper files.

TIGER is acknowledged as an external experimental backbone, not as code redistributed by this package. Its authors' work and applicable terms should be acknowledged separately when acquiring or using their implementation.

This scope statement is not a legal determination of ownership. Authors and the institution should review the selected code and license before publication.
