---
title: "6. AI for Transportation Infrastructure Management"
parent: Chapters
nav_order: 6
---

# Chapter 6: AI for Transportation Infrastructure Management
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 6` from the repository root; outputs go to `outputs/ch06/`.

## Case Study 1: Pavement deterioration and treatment screening

Simulate a 1,000-segment inventory, fit baseline / linear / ensemble models for one-year PCI change, rank features by permutation importance, and screen held-out segments for maintenance.

**Libraries:** scikit-learn, pandas, matplotlib

[Download `cs1_pavement_screening.py`](cs1_pavement_screening.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch06_infrastructure_management/cs1_pavement_screening.py){: .btn .fs-3 }

![Pavement deterioration and treatment screening]({{ '/docs/figures/chap06_fig01_pavement_screening.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_pavement_screening.py %}
```

## Case Study 2: Maintenance-trigger policies in a pavement digital twin

Simulate pavement condition over time and compare fixed-threshold maintenance triggers with a twin-lead policy via a parameter sweep of treatment rate, mean PCI, share below 70 and cost.

**Libraries:** NumPy, pandas, matplotlib

[Download `cs2_pavement_digital_twin.py`](cs2_pavement_digital_twin.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch06_infrastructure_management/cs2_pavement_digital_twin.py){: .btn .fs-3 }

![Maintenance-trigger policies in a pavement digital twin]({{ '/docs/figures/chap06_fig02_pavement_maintenance.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_pavement_digital_twin.py %}
```
