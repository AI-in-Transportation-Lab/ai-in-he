"""Chapter 7 - Case Study 1: Exposure-Adjusted Crash Analysis.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/07a.tex (4 listing block(s), concatenated in book order).
Run from the repository root with `make ch07` or `python scripts/run_all.py --chapter 7`;
figures are written to outputs/ch07/.
"""
import pandas as pd

spec = {
    "Clear": {"K": 1, "A": 2, "B": 5, "C": 8, "O": 14},
    "Rain":  {"K": 0, "A": 2, "B": 2, "C": 3, "O": 3},
    "Snow":  {"K": 0, "A": 1, "B": 2, "C": 2, "O": 2},
    "Fog":   {"K": 0, "A": 1, "B": 1, "C": 1, "O": 0},
}
rows = []
for weather, counts in spec.items():
    for severity, count in counts.items():
        for _ in range(count):
            rows.append({
                "crash_id": f"C{len(rows) + 1:03d}",
                "weather": weather,
                "severity": severity,
            })
crashes = pd.DataFrame(rows)
assert crashes.crash_id.is_unique
assert crashes.severity.isin(list("KABCO")).all()
crashes["ka"] = crashes.severity.isin(["K", "A"]).astype(int)
print(f"Crashes: {len(crashes)}; KA crashes: {crashes.ka.sum()}")


exposure = pd.DataFrame({
    "weather": ["Clear", "Rain", "Snow", "Fog"],
    "vehicle_miles": [3_000_000, 500_000, 200_000, 50_000],
})
summary = crashes.groupby("weather", sort=False).agg(
    crashes=("crash_id", "nunique"),
    ka_crashes=("ka", "sum"),
)
summary = summary.reset_index().merge(
    exposure, on="weather", validate="one_to_one"
)
assert summary.vehicle_miles.gt(0).all()
summary["ka_share_pct"] = (
    100 * summary.ka_crashes / summary.crashes
)
summary["crashes_per_mvm"] = (
    1_000_000 * summary.crashes / summary.vehicle_miles
)
summary["ka_per_mvm"] = (
    1_000_000 * summary.ka_crashes / summary.vehicle_miles
)
print(summary.round(2).to_string(index=False))


import math

def wilson(k, n, z=1.96):
    p = k / n
    centre = p + z**2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    denom = 1 + z**2 / n
    return (centre - half) / denom, (centre + half) / denom

for row in summary.itertuples():
    lo, hi = wilson(row.ka_crashes, row.crashes)
    print(f"{row.weather:<6} KA share {row.ka_share_pct:5.1f}%  "
          f"95% Wilson interval {100*lo:5.1f}% to {100*hi:5.1f}%")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOLD, SAGE, SKY = "#DAB400", "#78978D", "#6ADBF1"  # Moonrise Kingdom palette
RUST, OLIVE = "#CE6B08", "#A59D30"
INK, PAPER, GRID = "#3F4A45", "#FBF8F4", "#D5D5D3" # text, background, grid
plt.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SAGE,
    "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
x = list(range(len(summary)))
share = summary.ka_share_pct
ci = [wilson(k, n) for k, n in zip(summary.ka_crashes, summary.crashes)]
err = [[share[i] - 100 * ci[i][0] for i in x],
       [100 * ci[i][1] - share[i] for i in x]]
ax[0].errorbar(x, share, yerr=err, fmt="o", color=GOLD, ms=8,
               ecolor=SAGE, elinewidth=2, capsize=6)
for i, n in zip(x, summary.crashes):
    ax[0].text(i + 0.1, share[i], f"n={n}", fontsize=8, va="center")
ax[0].set(xticks=x, xticklabels=summary.weather, ylim=(0, 85),
          ylabel="KA share of crashes (%)",
          title="(a) KA share with 95% Wilson interval")
ax[1].bar(x, summary.ka_per_mvm, width=0.6, color=SAGE)
for i, v in zip(x, summary.ka_per_mvm):
    ax[1].text(i, v + 0.4, f"{v:.0f}", ha="center", fontsize=9)
ax[1].set(xticks=x, xticklabels=summary.weather,
          ylabel="KA crashes per million veh-mi",
          title="(b) Exposure-adjusted KA rate")
fig.tight_layout()
fig.savefig("chap07_fig01_ka_share_rate.png", dpi=200)
