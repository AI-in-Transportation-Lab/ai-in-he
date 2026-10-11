# data/

Cache for files the case studies need but that are not stored in git.

Run `python scripts/fetch_assets.py` to download them. At present only Chapter 4,
Case Study 2 needs external files:

| File | Source |
|---|---|
| `yolo11n.onnx` | <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx> |
| `traffic_scene.jpg` | <https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg> |

All other case studies generate their own synthetic data.
