# Methods and API

## Band Allocation

Let `F` be the number of one-sided STFT bins, `K` the total band count, and `K_l` the number of low-frequency single-bin bands. BA keeps the first `K_l` bins as individual bands. The high-frequency region has `F_h = F - K_l` bins and `K_h = K - K_l` bands.

For `K_h > 1`:

```text
q_j = (j - 1) / (K_h - 1),  j = 1, ..., K_h
a_j = exp(alpha * q_j)
b_j_target = F_h * a_j / sum_i(a_i)
```

The implementation floors the target widths and enforces a minimum of one bin. It then corrects the total:

1. If bins remain, add one bin per band from the highest-frequency band toward lower bands; repeat if necessary.
2. If too many bins have been assigned, remove one bin per eligible band from the lowest-frequency band upward. Only widths greater than one can be reduced; repeat if necessary.

For `K_h = 1`, the single high-frequency band receives all `F_h` bins. The result always contains `K` positive widths summing to `F`. Ordered cumulative sums define contiguous, non-overlapping boundaries covering every bin exactly once.

The correction is **not** a largest-fractional-remainder method. Do not replace it when reproducing the reported configuration. The real-valued target widths grow monotonically; the discrete correction is defined by the above procedure, and the package does not impose an additional sorting step. The default speech partition is nondecreasing.

### Functions

| Function | Return | Notes |
| --- | --- | --- |
| `build_exp_band(total_bins, num_bands, alpha=2.0)` | `list[int]` | Exponential allocation over the specified region |
| `build_hybrid_band(total_bins, low_bins=60, total_bands=67, alpha=2.0)` | `list[int]` | Complete BA partition |
| `band_slices(widths)` | `list[slice]` | Half-open bin intervals with zero-based indexing |

Counts must be integers, band widths must be positive, and `0 <= low_bins < total_bands <= total_bins`. `alpha` must be positive and finite. Zero low bins is supported as an exponential-only option, not as the paper's default. Numerically overflowing exponential settings are rejected rather than silently changed.

## Band Recalibration

Input `z` has shape `[B, K, D, T]`. For each batch item and band:

```text
p_k = mean_t(z_k[:, t])
g_k = sigmoid(W2 * relu(W1 * p_k + b1) + b2)
z_rec_k = g_k * z_k
```

The same MLP parameters are shared across bands. Temporal pooling removes only the time axis; the MLP maps the feature axis `D -> H -> 1`. Broadcasting restores the scalar gate across all features and frames of each band. There is no softmax normalization across bands, no Top-K mask, no residual addition, and no speaker-specific gate.

Use `BandRecalibration(feature_dim=128, hidden_dim=64)` from `abar_speech.recalibration`. `forward(z)` returns `[B, K, D, T]`; `forward(z, return_gate=True)` additionally returns `[B, K]` weights. Floating-point, nonempty, four-dimensional input is required. Place the module on the same device and use a compatible dtype as the input.

The parameter count is `D*H + H + H + 1`, or 8,321 for `D=128, H=64`. The module uses the default initialization of `torch.nn.Linear`. Mathematically, sigmoid weights lie in `(0, 1)`; floating-point saturation may reach the endpoints for extreme inputs.

`BandImportanceGate` is an alias for the historical standalone gate name. State dictionary keys remain `scorer.0.weight`, `scorer.0.bias`, `scorer.2.weight`, and `scorer.2.bias`. This preserves the gate's interface, not automatic compatibility with full-model checkpoints.
