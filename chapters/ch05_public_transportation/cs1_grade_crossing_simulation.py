"""Chapter 5 - Case Study 1: Discrete-Event Simulation of a Grade Crossing.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/05a.tex (2 listing block(s), concatenated in book order).
Run from the repository root with `make ch05` or `python scripts/run_all.py --chapter 5`;
figures are written to outputs/ch05/.
"""
import heapq
import numpy as np

ARRIVAL_RATE = 0.2    # vehicles per second (one every 5 s on average)
TRAIN_PERIOD = 300.0  # seconds between train arrivals
CLOSURE      = 30.0   # seconds the gates are down per train
SERVICE      = 2.0    # seconds for one vehicle to clear the crossing
HORIZON      = 3600.0 # one hour

def simulate(rate=ARRIVAL_RATE, seed=0, horizon=HORIZON, trace=False):
    """Discrete-event simulation of one crossing lane.
    Events: vehicle arrival, service completion, gate down, gate up.
    A vehicle already on the crossing completes; none starts while gates are down."""
    rng = np.random.default_rng(seed)
    events = []                                   # (time, order, kind)
    order = 0
    def push(t, kind):
        nonlocal order
        heapq.heappush(events, (t, order, kind)); order += 1
    push(rng.exponential(1 / rate), "arrive")
    t_train = TRAIN_PERIOD
    while t_train < horizon:
        push(t_train, "down"); push(t_train + CLOSURE, "up")
        t_train += TRAIN_PERIOD

    queue, busy, gates_down = [], False, False    # queue holds arrival times
    delays, trace_t, trace_q = [], [0.0], [0]
    q_area, t_prev = 0.0, 0.0
    def start_service(t):
        nonlocal busy
        arr = queue.pop(0)
        delays.append(t - arr)
        busy = True
        push(t + SERVICE, "depart")

    while events:
        t, _, kind = heapq.heappop(events)
        if t > horizon:
            break
        q_area += len(queue) * (t - t_prev); t_prev = t
        if kind == "arrive":
            queue.append(t)
            push(t + rng.exponential(1 / rate), "arrive")
        elif kind == "depart":
            busy = False
        elif kind == "down":
            gates_down = True
        elif kind == "up":
            gates_down = False
        if not busy and not gates_down and queue:
            start_service(t)
        if trace:
            trace_t.append(t); trace_q.append(len(queue))
    q_area += len(queue) * (horizon - t_prev)
    return dict(served=len(delays), mean_delay=float(np.mean(delays)),
                p95_delay=float(np.percentile(delays, 95)),
                mean_queue=q_area / horizon,
                max_queue=int(max(trace_q)) if trace else None,
                trace=(np.array(trace_t), np.array(trace_q)) if trace else None)

one = simulate(seed=1, trace=True)
print(f"Seed 1: served {one['served']}, mean delay {one['mean_delay']:.2f} s, "
      f"95th pct {one['p95_delay']:.1f} s, mean queue {one['mean_queue']:.2f}, "
      f"max queue {one['max_queue']}")

reps = [simulate(seed=s) for s in range(200)]
d = np.array([r["mean_delay"] for r in reps])
q = np.array([r["mean_queue"] for r in reps])
print(f"200 replications: mean delay {d.mean():.2f} s "
      f"(95% CI {d.mean() - 1.96 * d.std(ddof=1) / 200**0.5:.2f}"
      f"-{d.mean() + 1.96 * d.std(ddof=1) / 200**0.5:.2f}), "
      f"mean queue {q.mean():.2f} veh")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = 1 / SERVICE                                   # saturation flow, veh/s
def uniform_delay(rate, C=TRAIN_PERIOD, r=CLOSURE, s=S):
    """Deterministic (D/D/1) delay per vehicle for a periodic closure;
    identical in form to Webster's uniform-delay term with red time r."""
    x = rate / s
    return r**2 / (2 * C * (1 - x)) if rate * C < s * (C - r) else np.inf

def md1_wait(rate, s=S):
    """Mean M/D/1 queueing delay with no closures."""
    rho = rate / s
    return rho / (2 * s * (1 - rho))

capacity = S * (TRAIN_PERIOD - CLOSURE) / TRAIN_PERIOD  # veh/s
print(f"Capacity with closures: {capacity:.2f} veh/s")
print(f"{'rate':>5}{'sim':>8}{'D/D/1':>8}{'M/D/1':>8}{'sum':>8}")
rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
sim_d = []
for lam in rates:
    dd = np.mean([simulate(rate=lam, seed=s)["mean_delay"] for s in range(100)])
    sim_d.append(dd)
    print(f"{lam:>5.2f}{dd:>8.2f}{uniform_delay(lam):>8.2f}"
          f"{md1_wait(lam):>8.2f}{uniform_delay(lam) + md1_wait(lam):>8.2f}")

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
tt, qq = one["trace"]
ax[0].step(tt / 60, qq, where="post", lw=1.2, color=SAGE)
for k in range(1, int((HORIZON - 1) // TRAIN_PERIOD) + 1):
    t0 = k * TRAIN_PERIOD / 60
    ax[0].axvspan(t0, t0 + CLOSURE / 60, color=RUST, alpha=0.40, lw=0)
ax[0].set(xlabel="Time (min)", ylabel="Vehicles waiting",
          title="(a) Queue over one hour (shaded: gates down)")
ax[1].plot(rates, sim_d, "o-", color=GOLD, lw=2.2, ms=7,
           label="Simulation (100 runs)")
ax[1].plot(rates, [uniform_delay(l) + md1_wait(l) for l in rates], "s--",
           color=SAGE, lw=2, ms=6,
           label="D/D/1 closure + M/D/1 approximation")
ax[1].plot(rates, [uniform_delay(l) for l in rates], ":", color=SKY, lw=2.6,
           label="D/D/1 closure only")
ax[1].axvline(0.45, color=OLIVE, lw=1.4, ls="--")
ax[1].text(0.445, 1.5, "capacity\n0.45 veh/s", ha="right", fontsize=8)
ax[1].set(xlabel="Arrival rate (veh/s)", ylabel="Mean delay per vehicle (s)",
          title="(b) Delay versus demand")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("chap05_fig01_grade_crossing_queue.png", dpi=200)
