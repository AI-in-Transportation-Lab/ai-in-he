#!/usr/bin/env python3
"""Download the external files needed by Chapter 4, Case Study 2 into data/.

The URLs are the same ones listed in the header of the listing in the book.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ASSETS = {
    "yolo11n.onnx": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
    "traffic_scene.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    for name, url in ASSETS.items():
        dest = DATA_DIR / name
        if dest.exists() and not args.force:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name} <- {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"       {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
