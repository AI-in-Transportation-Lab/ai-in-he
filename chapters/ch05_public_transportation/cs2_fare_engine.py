"""Chapter 5 - Case Study 2: An Account-Based Fare Engine Prototype.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/05a.tex (2 listing block(s), concatenated in book order).
Run from the repository root with `make ch05` or `python scripts/run_all.py --chapter 5`;
figures are written to outputs/ch05/.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

FARES = {"Bus": 2.00, "Metro": 3.00, "Tram": 2.50}
DAILY_CAP = 10.00
TRANSFER_WINDOW = timedelta(minutes=30)
FREE_TRANSFERS = {("Bus", "Bus")}      # (previous paid mode, current mode)

@dataclass
class Tap:
    time: datetime
    mode: str

def price_day(taps):
    """Account-based fare calculation for one rider-day.
    Rules: flat fare by mode; free bus-to-bus transfer within 30 min of the
    last *paid* boarding; daily cap on the total charged."""
    ledger, total = [], 0.0
    last_paid = None                           # (time, mode) of last paid boarding
    for tap in sorted(taps, key=lambda x: x.time):
        base = FARES[tap.mode]
        reason = "full fare"
        if (last_paid and tap.time - last_paid[0] <= TRANSFER_WINDOW
                and (last_paid[1], tap.mode) in FREE_TRANSFERS):
            base, reason = 0.0, "free transfer"
        charge = round(min(base, max(0.0, DAILY_CAP - total)), 2)
        if charge < base:
            reason = "capped" if charge == 0 else "partially capped"
        total = round(total + charge, 2)
        if base > 0 and reason == "full fare":
            last_paid = (tap.time, tap.mode)
        ledger.append((tap.time.strftime("%H:%M"), tap.mode, base, charge,
                       total, reason))
    return ledger, total

def show(ledger):
    print(f"{'Time':<6}{'Mode':<7}{'Fare':>6}{'Charge':>8}{'Total':>7}  Rule")
    for t, m, b, c, tot, why in ledger:
        print(f"{t:<6}{m:<7}{b:>6.2f}{c:>8.2f}{tot:>7.2f}  {why}")

day = datetime(2025, 1, 1)
rider_a = [Tap(day.replace(hour=8), "Bus"),
           Tap(day.replace(hour=8, minute=20), "Metro"),
           Tap(day.replace(hour=13), "Bus"),
           Tap(day.replace(hour=18), "Metro"),
           Tap(day.replace(hour=18, minute=30), "Bus")]
ledger_a, total_a = price_day(rider_a)
show(ledger_a)
print(f"Rider A pays {total_a:.2f} "
      f"(pay-per-ride would be {sum(FARES[t.mode] for t in rider_a):.2f})\n")

# Rider B exercises the transfer window and a partial cap
rider_b = [Tap(day.replace(hour=7, minute=30), "Bus"),
           Tap(day.replace(hour=7, minute=55), "Bus"),    # 25 min: free
           Tap(day.replace(hour=8, minute=20), "Bus"),    # 50 min after paid: pays
           Tap(day.replace(hour=12), "Metro"),
           Tap(day.replace(hour=17), "Tram"),
           Tap(day.replace(hour=19), "Metro")]
ledger_b, total_b = price_day(rider_b)
show(ledger_b)
print(f"Rider B pays {total_b:.2f} "
      f"(pay-per-ride would be {sum(FARES[t.mode] for t in rider_b):.2f})")

# Regression tests for the fare rules
assert total_a == 10.00 and ledger_a[-1][3] == 0.00
assert [row[3] for row in ledger_b] == [2.00, 0.00, 2.00, 3.00, 2.50, 0.50]
assert all(row[4] <= DAILY_CAP for row in ledger_a + ledger_b)
print("All fare-rule tests passed.")


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
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for a, taps, led, name in [(ax[0], rider_a, ledger_a, "Rider A"),
                           (ax[1], rider_b, ledger_b, "Rider B")]:
    hrs = [t.time.hour + t.time.minute / 60 for t in taps]
    ppr = [sum(FARES[x.mode] for x in taps[:i + 1]) for i in range(len(taps))]
    a.step(hrs, ppr, where="post", ls="--", color=SAGE, lw=2, zorder=3,
           label="Pay per ride")                  # drawn on top: lines overlap
    a.step(hrs, [row[4] for row in led], where="post", color=GOLD, lw=2.2,
           label="Charged (ABT rules)")
    a.axhline(DAILY_CAP, color=RUST, lw=1.4, ls=":", label="Daily cap")
    a.set(xlabel="Time of day (h)", title=name)
    a.legend(fontsize=8, loc="upper left", frameon=False)
ax[0].set_ylabel("Cumulative fare ($)")
fig.tight_layout()
fig.savefig("chap05_fig02_fare_ledger.png", dpi=200)
