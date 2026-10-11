---
title: "4. AI for Connected and Automated Vehicles"
parent: Chapters
nav_order: 4
---

# Chapter 4: AI for Connected and Automated Vehicles
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 4` from the repository root; outputs go to `outputs/ch04/`.

## Case Study 1: A synthetic, automatically labeled perception dataset

Generate 200 top-down road scenes with controlled vehicle class, density, truncation and visibility; export YOLO and JSON labels; audit the labels programmatically. Writes `synthetic_*.png/.txt` and `annotations.json`.

**Libraries:** OpenCV, NumPy, matplotlib

[Download cs1_synthetic_perception_dataset.py](cs1_synthetic_perception_dataset.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch04_connected_automated_vehicles/cs1_synthetic_perception_dataset.py){: .btn .fs-3 }

![A synthetic, automatically labeled perception dataset]({{ '/docs/figures/chap04_fig_synthetic_samples.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_synthetic_perception_dataset.py %}
```

## Case Study 2: Edge object detection and robustness testing

Run a pretrained YOLO11n ONNX detector on CPU, implement letterbox pre-/post-processing with NMS, and measure how detections change under fog, low light, sensor noise and motion blur. **Needs `yolo11n.onnx` and `traffic_scene.jpg`: run `python scripts/fetch_assets.py` first.**

**Libraries:** ONNX Runtime, OpenCV, matplotlib

[Download cs2_onnx_detector_robustness.py](cs2_onnx_detector_robustness.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch04_connected_automated_vehicles/cs2_onnx_detector_robustness.py){: .btn .fs-3 }

![Edge object detection and robustness testing]({{ '/docs/figures/chap04_fig_detection_robustness.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_onnx_detector_robustness.py %}
```
