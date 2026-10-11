---
title: "7. AI-Powered Data Analytics and Decision Support"
parent: Chapters
nav_order: 7
---

# Chapter 7: AI-Powered Data Analytics and Decision Support
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 7` from the repository root; outputs go to `outputs/ch07/`.

## Case Study 1: Exposure-adjusted crash analysis

From record-level synthetic crashes (KABCO severity), compute conditional severity proportions, exposure-adjusted rates and Wilson intervals by weather condition.

**Libraries:** pandas, NumPy, matplotlib

[Download `cs1_exposure_adjusted_crashes.py`](cs1_exposure_adjusted_crashes.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch07_analytics_decision_support/cs1_exposure_adjusted_crashes.py){: .btn .fs-3 }

![Exposure-adjusted crash analysis]({{ '/docs/figures/chap07_fig01_ka_share_rate.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_exposure_adjusted_crashes.py %}
```

## Case Study 2: Evidence screening in crash narratives

Distinguish a documented assertion from the mere presence of a word in ten invented narratives (affirmed / negated / unresolved / not mentioned) and report precision, recall and agreement.

**Libraries:** re, matplotlib

[Download `cs2_crash_narrative_screening.py`](cs2_crash_narrative_screening.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/ai-in-he/blob/main/chapters/ch07_analytics_decision_support/cs2_crash_narrative_screening.py){: .btn .fs-3 }

![Evidence screening in crash narratives]({{ '/docs/figures/chap07_fig02_narrative_screen.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_crash_narrative_screening.py %}
```
