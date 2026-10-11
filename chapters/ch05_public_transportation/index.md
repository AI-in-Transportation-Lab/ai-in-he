---
title: "5. AI in Public Transportation and Mobility Services"
parent: Chapters
nav_order: 5
---

# Chapter 5: AI in Public Transportation and Mobility Services
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 5` from the repository root; outputs go to `outputs/ch05/`.

## Case Study 1: Discrete-event simulation of a grade crossing

Estimate road-user delay at a single-lane approach to a highway-rail grade crossing; compare an event-queue (heapq) discrete-event simulation with D/D/1 and M/D/1 analytical approximations and show how delay grows as demand approaches capacity.

**Libraries:** NumPy, heapq, matplotlib

[Download `cs1_grade_crossing_simulation.py`](cs1_grade_crossing_simulation.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch05_public_transportation/cs1_grade_crossing_simulation.py){: .btn .fs-3 }

![Discrete-event simulation of a grade crossing]({{ '/docs/figures/chap05_fig01_grade_crossing_queue.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_grade_crossing_simulation.py %}
```

## Case Study 2: An account-based fare engine prototype

Implement the back-office pricing logic of an account-based ticketing system for one rider-day, verify it against hand-computed results, and compare the amount charged with pay-per-ride fares.

**Libraries:** dataclasses, matplotlib

[Download `cs2_fare_engine.py`](cs2_fare_engine.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch05_public_transportation/cs2_fare_engine.py){: .btn .fs-3 }

![An account-based fare engine prototype]({{ '/docs/figures/chap05_fig02_fare_ledger.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_fare_engine.py %}
```
