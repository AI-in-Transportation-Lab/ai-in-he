"""Chapter 6 - Case Study 2: Maintenance-Trigger Policies in a Simulated Pavement Digital Twin.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/06a.tex (2 listing block(s), concatenated in book order).
Run from the repository root with `make ch06` or `python scripts/run_all.py --chapter 6`;
figures are written to outputs/ch06/.
"""
import numpy as np

MONTHS    = 120      # 10-year horizon
THRESHOLD = 70.0     # minimum acceptable PCI
RESET_PCI = 95.0     # condition after treatment
OBS_SD    = 1.5      # s.d. of measurement error on monthly observations
WINDOW    = 6        # observations used by the twin to estimate the trend

def monthly_loss(pci, k, rng):
    """True monthly PCI loss: k*0.8 when new, accelerating as PCI falls."""
    return max(0.0, k * 0.8 * (1 + (100 - pci) / 50) + rng.normal(0, 0.25))

def simulate(policy, param, seed):
    """policy = 'threshold' (param = trigger PCI) or 'twin' (param = lead, months)."""
    rng_true = np.random.default_rng(seed)           # identical deterioration path
    rng_obs  = np.random.default_rng(seed + 10_000)  # identical sensor noise
    k = rng_true.lognormal(mean=0.0, sigma=0.5)      # segment-specific rate multiplier
    pci, hist, obs, treated_at = 100.0, [], [], []
    for m in range(MONTHS):
        pci -= monthly_loss(pci, k, rng_true)
        hist.append(pci)                             # true condition this month
        y = pci + rng_obs.normal(0, OBS_SD)          # observed condition
        obs.append(y)
        if policy == "threshold":
            treat = y < param
        else:                                        # forecast-based trigger
            treat = y < THRESHOLD
            if len(obs) >= WINDOW:
                slope, level = np.polyfit(np.arange(WINDOW), obs[-WINDOW:], 1)
                treat |= level + slope * (WINDOW - 1 + param) < THRESHOLD
        if treat:
            treated_at.append(pci)
            pci, obs = RESET_PCI, []
    return np.array(hist), np.array(treated_at)

def agency_cost(treated_at, convex):
    """Cost per treatment (arbitrary units): fixed mobilisation + restoration."""
    d = RESET_PCI - treated_at
    return float(np.sum(10 + (0.04 * d ** 2 if convex else d)))

def evaluate(policy, param, n_runs=500):
    out = []
    for seed in range(n_runs):
        h, t = simulate(policy, param, seed)
        out.append([len(t), h.mean(), h.min(), 100 * np.mean(h < THRESHOLD),
                    agency_cost(t, False), agency_cost(t, True)])
    return np.mean(out, axis=0)

cases = [("threshold", 70, "Reactive (PCI<70)"),
         ("threshold", 75, "Fixed trigger (PCI<75)"),
         ("twin", 3, "Twin, 3-month lead"),
         ("twin", 6, "Twin, 6-month lead")]
print(f"{'Policy':<24}{'Treat':>6}{'MeanPCI':>9}{'MinPCI':>8}{'%<70':>6}"
      f"{'CostLin':>9}{'CostCvx':>9}")
for pol, par, label in cases:
    r = evaluate(pol, par)
    print(f"{label:<24}{r[0]:>6.1f}{r[1]:>9.1f}{r[2]:>8.1f}{r[3]:>6.1f}"
          f"{r[4]:>9.0f}{r[5]:>9.0f}")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

thr = [(t, evaluate("threshold", t)) for t in range(70, 81, 2)]
twn = [(L, evaluate("twin", L)) for L in range(0, 9, 2)]

GOLD, SAGE, SKY = "#DAB400", "#78978D", "#6ADBF1"  # Moonrise Kingdom palette
RUST, OLIVE = "#CE6B08", "#A59D30"
INK, PAPER, GRID = "#3F4A45", "#FBF8F4", "#D5D5D3" # text, background, grid
plt.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SAGE,
    "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
h_r, _ = simulate("threshold", 70, seed=3)
h_t, _ = simulate("twin", 6, seed=3)
ax[0].plot(h_r, color=GOLD, lw=1.8, label="Reactive (PCI<70)")
ax[0].plot(h_t, color=SAGE, lw=1.8, label="Twin, 6-month lead")
ax[0].axhline(THRESHOLD, ls="--", color=RUST, lw=1.4, label="PCI 70")
ax[0].set(xlabel="Month", ylabel="True PCI", title="(a) One segment, seed 3")
ax[0].legend(fontsize=8, frameon=False)

ax[1].plot([r[0] for _, r in thr], [r[1] for _, r in thr], "o-",
           color=GOLD, lw=2.2, ms=7, label="Fixed trigger 70-80")
ax[1].plot([r[0] for _, r in twn], [r[1] for _, r in twn], "s-",
           color=SAGE, lw=2, ms=6, label="Twin, lead 0-8 months")
ax[1].set(xlabel="Treatments per 10 years", ylabel="Mean PCI",
          title="(b) Condition vs. treatment frequency (500 runs)")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("chap06_fig02_pavement_maintenance.png", dpi=200)

print("\nFixed trigger  Treat  MeanPCI  %<70  CostCvx")
for t, r in thr:
    print(f"{t:>13}{r[0]:>7.1f}{r[1]:>9.1f}{r[3]:>6.1f}{r[5]:>9.0f}")
print("Twin lead (mo) Treat  MeanPCI  %<70  CostCvx")
for L, r in twn:
    print(f"{L:>13}{r[0]:>7.1f}{r[1]:>9.1f}{r[3]:>6.1f}{r[5]:>9.0f}")
