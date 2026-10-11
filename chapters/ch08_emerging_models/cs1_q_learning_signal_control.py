"""Chapter 8 - Case Study 1: Learning a Signal Control Policy from Experience.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/08a.tex (2 listing block(s), concatenated in book order).
Run from the repository root with `make ch08` or `python scripts/run_all.py --chapter 8`;
figures are written to outputs/ch08/.
"""
import numpy as np

rng = np.random.default_rng(2026)

# ----- Simulation parameters -----
P_NS, P_EW = 0.4, 0.2      # arrival probability per step on each approach
MAX_Q      = 5             # queue cap used for the tabular state
STEPS      = 50            # time steps per episode
EPISODES   = 5000
ALPHA, GAMMA = 0.1, 0.9
EPS, EPS_MIN, EPS_DECAY = 1.0, 0.05, 0.999

def step(ns, ew, action):
    """Arrivals, then one departure from the approach that has green."""
    ns = min(ns + (rng.random() < P_NS), MAX_Q)
    ew = min(ew + (rng.random() < P_EW), MAX_Q)
    if action == 0 and ns > 0:
        ns -= 1
    elif action == 1 and ew > 0:
        ew -= 1
    return ns, ew, -(ns + ew)          # reward = negative total queue

# ----- Q-learning -----
Q = np.zeros((MAX_Q + 1, MAX_Q + 1, 2))
for ep in range(EPISODES):
    ns, ew = 0, 0
    for t in range(STEPS):
        if rng.random() < EPS:
            a = int(rng.integers(2))
        else:
            a = int(np.argmax(Q[ns, ew]))
        ns2, ew2, r = step(ns, ew, a)
        # update the value of the state-action pair that was actually taken
        target = r + GAMMA * np.max(Q[ns2, ew2])
        Q[ns, ew, a] += ALPHA * (target - Q[ns, ew, a])
        ns, ew = ns2, ew2
    EPS = max(EPS_MIN, EPS * EPS_DECAY)

# ----- Evaluation against two baselines -----
def run_episode(policy):
    ns, ew, total_wait = 0, 0, 0
    for t in range(STEPS):
        total_wait += ns + ew
        if policy == "rl":
            a = int(np.argmax(Q[ns, ew]))
        elif policy == "fixed":           # strict alternation
            a = t % 2
        else:                              # "actuated": serve longer queue
            a = 0 if ns >= ew else 1
        ns, ew, _ = step(ns, ew, a)
    return total_wait

RUNS = 2000
results = {}  # waits per policy, for plotting
for pol in ["fixed", "actuated", "rl"]:
    waits = [run_episode(pol) for _ in range(RUNS)]
    results[pol] = waits
    print(f"{pol:>9}: mean total queued vehicle-steps = {np.mean(waits):5.1f} "
          f"(sd {np.std(waits):4.1f})")

# Inspect the learned policy (rows = NS queue, cols = EW queue; 0 = NS green)
print("Greedy policy (rows NS queue 0-5, cols EW queue 0-5):")
print(np.argmax(Q, axis=2))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
names = ["fixed", "actuated", "rl"]
means = [np.mean(results[p]) for p in names]
sds = [np.std(results[p]) for p in names]
ax[0].bar(names, means, yerr=sds, capsize=6, width=0.6,
          color=[SAGE, SKY, GOLD], ecolor=INK)
for i, m in enumerate(means):
    ax[0].text(i + 0.32, m, f"{m:.1f}", va="center", fontsize=9)
ax[0].set(ylabel="Queued vehicle-steps per episode",
          title=f"(a) Mean $\\pm$ sd over {RUNS} episodes")

policy = np.argmax(Q, axis=2)                  # 0 = NS green, 1 = EW green
ax[1].imshow(policy, cmap=ListedColormap([GOLD, SKY]), origin="lower")
for i in range(MAX_Q + 1):
    for j in range(MAX_Q + 1):
        ax[1].text(j, i, "NS" if policy[i, j] == 0 else "EW",
                   ha="center", va="center", fontsize=8)
ax[1].set(xlabel="EW queue", ylabel="NS queue",
          title="(b) Learned greedy policy (green phase)")
ax[1].grid(False)
fig.tight_layout()
fig.savefig("chap08_fig01_signal_control.png", dpi=200)
