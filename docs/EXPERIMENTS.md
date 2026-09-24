# Experimental Settings and Result Provenance

The experiments were retrained after revising BR. This release isolates BA and BR; it does not provide the full system, training logs, model weights, or new numerical separation results. The settings below were read from the supplied research configuration, not inferred from a completed run.

## Speech Module Settings

| Setting | Value |
| --- | --- |
| Sample rate / sources | 16 kHz / 2 |
| STFT window/FFT / hop | 640 / 160 samples |
| One-sided bins / total bands | 321 / 67 |
| Single-bin low-frequency bands | 60 |
| High-frequency bands / growth parameter | 7 / 2.0 |
| Encoded feature dimension / BR hidden dimension | 128 / 64 |
| BR activation / output | SiLU / `2 * sigmoid(logits)` |

The external backbone configuration uses `out_channels=128`, `in_channels=256`, `num_blocks=4`, and `upsampling_depth=5`. These are integration settings, not backbone code provided by this repository.

## EchoSet Training Configuration

The inspected full-data configuration specifies:

- Random 3-s training segments, batch size 1, eight data-loader workers, and pinned memory.
- Permutation-invariant pairwise negative SNR for training and pairwise negative SI-SDR for validation.
- Adam with initial learning rate `0.001` and zero weight decay.
- A plateau learning-rate scheduler with patience 10 and factor 0.5.
- Early stopping with patience 20 and a maximum budget of 200 epochs.

The source also contains a separate small-data configuration with a 100-epoch maximum, zero data-loader workers, and no pinned memory. It is not the full-data experiment. Both configuration files leave the checkpoint path unset; this alone does not establish whether a particular execution started from scratch or used runtime overrides.

Configured epoch limits are not completed epoch counts. The inspection does not establish final scores, training hardware, seeds, effective runtime overrides, or completion of retraining on every speech dataset. Report those details from the corresponding experiment records.

## Reporting Results

Associate each reported measurement with its dataset and split, effective configuration, checkpoint, evaluation protocol, and the exact BR computation. Do not carry numerical scores from earlier ReLU/unscaled-sigmoid experiments into the current implementation without evaluating the matching retrained model.

The public name remains **BR**. Earlier and current implementations share parameter names and shapes, so successful checkpoint loading does not identify the computation used during training. Keep this provenance in experiment records even when no development-version label is used in the paper.

For manuscript consistency, the BR equation and any expanded MLP diagram should use SiLU and `g_k = 2 * sigmoid(u_k)`, with mathematical range `(0, 2)`. The initialization description should state zero biases and small random final-layer weights. Any gate visualizations or separation results should come from the corresponding retrained checkpoint.
