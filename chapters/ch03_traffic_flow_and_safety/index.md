---
title: "3. AI for Traffic Flow, Congestion, and Road Safety"
parent: Chapters
nav_order: 3
---

# Chapter 3: AI for Traffic Flow, Congestion, and Road Safety
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 3` from the repository root; outputs go to `outputs/ch03/`.

## Case Study 1: Short-term traffic volume forecasting

Forecast hourly volume 1 h and 24 h ahead; compare seasonal-naive, SARIMA, gradient boosting with calendar/weather features, and an LSTM, with errors reported overall and on holidays. Also writes `synthetic_hourly_volume.csv`.

**Libraries:** statsmodels, TensorFlow/Keras, scikit-learn

[Download `cs1_volume_forecasting.py`](cs1_volume_forecasting.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch03_traffic_flow_and_safety/cs1_volume_forecasting.py){: .btn .fs-3 }

![Short-term traffic volume forecasting]({{ '/docs/figures/chap03_fig01_traffic_forecast.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_volume_forecasting.py %}
```

## Case Study 2: Real-time crash risk classification

Estimate the probability of a crash in a 5-minute freeway interval; compare logistic regression, gradient boosting and a neural network; set the alert threshold from an operational alert budget.

**Libraries:** TensorFlow/Keras, scikit-learn

[Download `cs2_crash_risk_classification.py`](cs2_crash_risk_classification.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch03_traffic_flow_and_safety/cs2_crash_risk_classification.py){: .btn .fs-3 }

![Real-time crash risk classification]({{ '/docs/figures/chap03_fig02_crash_risk.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_crash_risk_classification.py %}
```
