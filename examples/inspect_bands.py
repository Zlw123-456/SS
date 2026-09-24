"""Inspect allocation widths and intervals; no audio or backbone is needed."""

import argparse
import json
from pathlib import Path

from abar_speech import band_slices, build_hybrid_band


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "speech_ba_br.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    widths = build_hybrid_band(**config["ba"])
    print(f"Number of bands: {len(widths)}")
    print(f"Covered frequency bins: {sum(widths)}")
    print(f"Widths: {widths}")
    print("Band | start bin (inclusive) | stop bin (exclusive) | width")
    for index, (region, width) in enumerate(zip(band_slices(widths), widths)):
        print(f"{index:4d} | {region.start:21d} | {region.stop:20d} | {width:5d}")


if __name__ == "__main__":
    main()
