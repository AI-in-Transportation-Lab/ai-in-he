---
title: Home
layout: home
nav_order: 1
permalink: /
---

# AI in Highway Engineering
{: .fs-9 }

Runnable code for every case study in the book *Artificial Intelligence in Highway Engineering* by Subasish Das.
{: .fs-6 .fw-300 }

[Browse the chapters]({{ '/chapters/' | relative_url }}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he){: .btn .fs-5 .mb-4 .mb-md-0 }

---

<p align="center">
  <img src="{{ '/docs/figures/chap06_fig01_pavement_screening.png' | relative_url }}" width="32%">
  <img src="{{ '/docs/figures/chap03_fig01_traffic_forecast.png' | relative_url }}" width="32%">
  <img src="{{ '/docs/figures/chap08_fig02_charger_qubo.png' | relative_url }}" width="32%">
</p>

Every code listing in the book's case studies is here as a runnable Python script, organised by chapter, with the code shown on each chapter page. The scripts generate their own synthetic data, so you can reproduce each figure and printed result without data access agreements. Chapter 4, Case Study 2 downloads one small model and one image.

## Chapters

Chapter 1 is conceptual and has no code listings.

| Ch. | Title | Case studies |
|---|---|---|
| 2 | [AI Core Concepts and Methodologies]({{ '/chapters/ch02_core_concepts/' | relative_url }}) | A neural network for hourly traffic volume; A graph neural network for volume on uncounted local roads |
| 3 | [AI for Traffic Flow, Congestion, and Road Safety]({{ '/chapters/ch03_traffic_flow_and_safety/' | relative_url }}) | Short-term traffic volume forecasting; Real-time crash risk classification |
| 4 | [AI for Connected and Automated Vehicles]({{ '/chapters/ch04_connected_automated_vehicles/' | relative_url }}) | A synthetic, automatically labeled perception dataset; Edge object detection and robustness testing |
| 5 | [AI in Public Transportation and Mobility Services]({{ '/chapters/ch05_public_transportation/' | relative_url }}) | Discrete-event simulation of a grade crossing; An account-based fare engine prototype |
| 6 | [AI for Transportation Infrastructure Management]({{ '/chapters/ch06_infrastructure_management/' | relative_url }}) | Pavement deterioration and treatment screening; Maintenance-trigger policies in a pavement digital twin |
| 7 | [AI-Powered Data Analytics and Decision Support]({{ '/chapters/ch07_analytics_decision_support/' | relative_url }}) | Exposure-adjusted crash analysis; Evidence screening in crash narratives |
| 8 | [Emerging AI Models and Future Horizons in Highway Engineering]({{ '/chapters/ch08_emerging_models/' | relative_url }}) | Learning a signal control policy from experience; Formulating charger siting as a QUBO |

## A taste of the code

Simulated annealing for charger siting (Chapter 8, Case Study 2):

```python
def anneal(Q, c, sweeps=3000, T0=50.0, T1=0.05, seed=0):
    r = np.random.default_rng(seed)
    x = r.integers(0, 2, len(Q)).astype(float)
    e = energy(x, Q, c)
    for s in range(sweeps):
        T = T0 * (T1 / T0) ** (s / sweeps)
        i = r.integers(len(Q))
        x[i] = 1 - x[i]
        e_new = energy(x, Q, c)
        if e_new <= e or r.random() < np.exp(-(e_new - e) / T):
            e = e_new
        else:
            x[i] = 1 - x[i]
    return e, x
```

## Quick start

See [Getting started]({{ '/getting-started/' | relative_url }}) for details.

```bash
git clone https://github.com/AI-in-Transportation-Lab/ai-in-he.git
cd ai-in-he
uv sync
uv run python scripts/run_all.py
```

{% comment %}
TODO: restore once the final book citation is available.

## Citation

```bibtex
@book{das2026aihe,
  author = {Das, Subasish},
  title  = {Artificial Intelligence in Highway Engineering},
  year   = {2026}
}
```
{% endcomment %}

Developed with the Artificial Intelligence in Transportation (AIT) Lab at Texas State University.
