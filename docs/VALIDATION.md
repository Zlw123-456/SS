# Validation Notes

## Locally checked

The updated package was checked locally on Windows with Python 3.12.14, NumPy 2.3.5, and PyTorch 2.5.1+cpu.

- All 27 unit tests passed: 12 allocation tests and 15 recalibration tests.
- BR checks include SiLU/scaled-sigmoid computation, attenuation and amplification, exact initialization, identity for a zeroed scorer, and the limits of state-dictionary compatibility.
- The 12 allocation tests also passed in a Python environment without PyTorch installed.
- Editable package installation succeeded.
- Both documented examples completed successfully.
- The default allocation produced 67 widths summing to 321, with high-band widths `[11, 15, 21, 30, 43, 59, 82]`.
- The BR example preserved `[2, 67, 128, 20]`, returned `[2, 67]` gates, reported 8,321 parameters, and produced finite input gradients.

## Comparison with the research additions

A separate local verification extracted only the two allocation functions and standalone gate definition from the research source. It did not import the backbone or copy the original combined model into this repository.

- For 2,001 parameter settings, the extracted release matched the original exponential and hybrid allocation outputs exactly: 4,002 list comparisons.
- Across two feature/hidden-dimension settings, two input shapes per setting, and three random seeds (12 shape/seed cases), initialization, outputs, returned gates, input gradients, and parameter gradients matched the current research implementation exactly under the checked CPU environment.

The allocation comparison used a fixed random seed and included the default speech partition. The BR comparison checked seeds 1, 11, and 123. It supports preservation of the checked module behavior; it is not a guarantee for every floating-point environment or extreme parameter setting.

## Not claimed

- No full speech model was trained or evaluated as part of this package check.
- Dataset loading, checkpoint-to-model reproduction, and SDRi/SI-SNRi reproduction are outside this release.
- GPU/CUDA execution was not tested.
- The configured GitHub Actions jobs have not run remotely; local tests do not imply a green GitHub workflow.
- Python 3.10 and other allowed dependency versions were not tested locally. The CI configuration includes Python 3.10 and 3.12 for future checks.
- No performance speedup, end-to-end quality result, or music-separation result is established by these module tests.
