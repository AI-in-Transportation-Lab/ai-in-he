---
title: "2. AI Core Concepts and Methodologies"
parent: Chapters
nav_order: 2
---

# Chapter 2: AI Core Concepts and Methodologies
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 2` from the repository root; outputs go to `outputs/ch02/`.

## Case Study 1: A neural network for hourly traffic volume

Predict hourly volume from calendar and weather covariates with a multilayer perceptron; compare it with a seasonal-naive baseline, a linear model with interactions, and gradient boosting; inspect what the network learned (permutation importance).

**Libraries:** scikit-learn, pandas, matplotlib

[Download `cs1_mlp_traffic_volume.py`](cs1_mlp_traffic_volume.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch02_core_concepts/cs1_mlp_traffic_volume.py){: .btn .fs-3 }

![A neural network for hourly traffic volume]({{ '/docs/figures/chap02_fig01_mlp_traffic.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_mlp_traffic_volume.py %}
```

## Case Study 2: A graph neural network for volume on uncounted local roads

Estimate AADT on uncounted segments of a synthetic road network from segment attributes and network structure; compare a class-median lookup, gradient boosting, and a Keras graph convolutional network (GCN) trained on counted segments only.

**Libraries:** TensorFlow/Keras, networkx, scikit-learn

[Download `cs2_gnn_local_volume.py`](cs2_gnn_local_volume.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch02_core_concepts/cs2_gnn_local_volume.py){: .btn .fs-3 }

![A graph neural network for volume on uncounted local roads]({{ '/docs/figures/chap02_fig02_gnn_volume.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_gnn_local_volume.py %}
```
