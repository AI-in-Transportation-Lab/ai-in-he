---
title: "8. Emerging AI Models and Future Horizons in Highway Engineering"
parent: Chapters
nav_order: 8
---

# Chapter 8: Emerging AI Models and Future Horizons in Highway Engineering
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 8` from the repository root; outputs go to `outputs/ch08/`.

## Case Study 1: Learning a signal control policy from experience

Train a tabular Q-learning controller for a two-phase intersection and compare it with a fixed-time (strict alternation) and an actuated (serve the longer queue) baseline over repeated episodes.

**Libraries:** NumPy, matplotlib

[Download cs1_q_learning_signal_control.py](cs1_q_learning_signal_control.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch08_emerging_models/cs1_q_learning_signal_control.py){: .btn .fs-3 }

![Learning a signal control policy from experience]({{ '/docs/figures/chap08_fig01_signal_control.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_q_learning_signal_control.py %}
```

## Case Study 2: Formulating charger siting as a QUBO

Cast EV-charger siting as a QUBO, solve it by exhaustive search and simulated annealing, and compare both with a top-demand rule.

**Libraries:** NumPy, itertools, matplotlib

[Download cs2_charger_qubo.py](cs2_charger_qubo.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch08_emerging_models/cs2_charger_qubo.py){: .btn .fs-3 }

![Formulating charger siting as a QUBO]({{ '/docs/figures/chap08_fig02_charger_qubo.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_charger_qubo.py %}
```
