"""Chapter 8 - Case Study 2: Formulating Charger Siting as a QUBO.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/08a.tex (2 listing block(s), concatenated in book order).
Run from the repository root with `make ch08` or `python scripts/run_all.py --chapter 8`;
figures are written to outputs/ch08/.
"""
import itertools
import numpy as np

def build_qubo(demand, overlap, k, P):
    """Minimise  -sum d_i x_i + sum_{i<j} o_ij x_i x_j + P (sum x_i - k)^2
    returned as an upper-triangular matrix Q with f(x) = x^T Q x + const."""
    n = len(demand)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -demand[i] + P * (1 - 2 * k)       # linear terms (x_i^2 = x_i)
        for j in range(i + 1, n):
            Q[i, j] = overlap[i, j] + 2 * P          # quadratic terms
    return Q, P * k**2

def energy(x, Q, const):
    return x @ Q @ x + const

# ---------- Part A: the three-site toy problem ----------
demand = np.array([5.0, 3.0, 4.0])
Q3, c3 = build_qubo(demand, np.zeros((3, 3)), k=2, P=10.0)
print("QUBO matrix (3 sites):\n", Q3)
for bits in itertools.product([0, 1], repeat=3):
    x = np.array(bits)
    print(bits, f"energy = {energy(x, Q3, c3):6.1f}")

# ---------- Part B: 20 candidate sites with overlapping coverage ----------
rng = np.random.default_rng(7)
n, k, P = 20, 5, 40.0
pos = np.sort(rng.uniform(0, 200, n))                 # milepost of each site (mi)
demand = rng.uniform(20, 60, n).round(1)              # EV trips served per hour
dist = np.abs(pos[:, None] - pos[None, :])
overlap = np.where(dist < 15, 0.5 * np.minimum.outer(demand, demand)
                   * (1 - dist / 15), 0.0)            # double-counted demand
np.fill_diagonal(overlap, 0)
Q, c = build_qubo(demand, overlap, k, P)

# Exhaustive search over all 2^20 bitstrings (the "ground truth")
best_e, best_x = np.inf, None
for bits in itertools.product([0, 1], repeat=n):
    x = np.fromiter(bits, dtype=float, count=n)
    e = energy(x, Q, c)
    if e < best_e:
        best_e, best_x = e, x

# Classical simulated annealing: a quantum-inspired heuristic stand-in
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

results = [anneal(Q, c, seed=s) for s in range(20)]
hits = sum(abs(e - best_e) < 1e-9 for e, _ in results)
print(f"\nExhaustive optimum: energy {best_e:.1f}, sites {np.flatnonzero(best_x)}")
print(f"Annealing found the optimum in {hits} of 20 restarts; "
      f"best annealed energy {min(e for e, _ in results):.1f}")

# Naive rule: pick the five highest-demand sites, ignoring overlap
greedy = np.zeros(n); greedy[np.argsort(demand)[-k:]] = 1
print(f"Top-{k}-demand rule: energy {energy(greedy, Q, c):.1f}, "
      f"sites {np.flatnonzero(greedy)}")


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
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8),
                       gridspec_kw={"width_ratios": [1.6, 1]})
ax[0].vlines(pos, 0, demand, color=SAGE, lw=2)
ax[0].scatter(pos, demand, s=30, color=SAGE, label="Candidate site")
opt, top = best_x == 1, greedy == 1
ax[0].scatter(pos[opt], demand[opt], s=150, facecolor="none",
              edgecolor=GOLD, lw=2.5, label="QUBO optimum")
ax[0].scatter(pos[top & ~opt], demand[top & ~opt], s=80, marker="X",
              color=RUST, label="Top-5 rule only")
ax[0].set(xlabel="Milepost (mi)", ylabel="EV trips served per hour",
          title="(a) Candidate sites along the corridor")
ax[0].legend(fontsize=8, frameon=False, loc="upper left")

e_runs = sorted(e for e, _ in results)
ax[1].plot(range(1, 21), e_runs, "o", color=SKY, ms=7,
           label="Annealing restarts")
ax[1].axhline(best_e, color=GOLD, lw=2, label="Exhaustive optimum")
ax[1].axhline(energy(greedy, Q, c), color=RUST, ls="--", lw=1.4,
              label="Top-5 rule")
ax[1].set(xlabel="Restart (sorted by energy)", ylabel="QUBO energy",
          title="(b) Annealing restarts")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("chap08_fig02_charger_qubo.png", dpi=200)
