"""Run a synthetic forward/backward example, not a speech-separation model."""

import argparse
import json
from pathlib import Path

import torch

from abar_speech import build_hybrid_band
from abar_speech.recalibration import BandRecalibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "speech_ba_br.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    widths = build_hybrid_band(**config["ba"])
    torch.manual_seed(0)
    gate = BandRecalibration(**config["br"])
    x = torch.randn(2, len(widths), config["br"]["feature_dim"], 20, requires_grad=True)
    weighted, scores = gate(x, return_gate=True)
    weighted.square().mean().backward()
    print("Synthetic feature demo only; no backbone or separation result.")
    print(f"Input/output shapes: {tuple(x.shape)} / {tuple(weighted.shape)}")
    print(f"Gate shape: {tuple(scores.shape)}")
    print(f"BR parameters: {sum(parameter.numel() for parameter in gate.parameters()):,}")
    print(f"Finite input gradients: {bool(torch.isfinite(x.grad).all())}")


if __name__ == "__main__":
    main()
