"""Chapter 6 - Case Study 1: Predicting One-Year Pavement Deterioration and Screening Segments for Treatment.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/06a.tex (4 listing block(s), concatenated in book order).
Run from the repository root with `make ch06` or `python scripts/run_all.py --chapter 6`;
figures are written to outputs/ch06/.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 1000

aadt   = rng.uniform(1_000, 20_000, n)          # annual average daily traffic
truck  = rng.uniform(0.03, 0.25, n)             # truck share of AADT
age    = rng.uniform(1, 30, n)                  # years since last major treatment
pci    = rng.uniform(55, 100, n)                # current PCI (0-100)
ftc    = rng.integers(0, 80, n)                 # annual freeze-thaw cycles

trucks_k = aadt * truck / 1_000                 # thousands of trucks per day

# Assumed (illustrative) one-year PCI loss: accelerates as condition worsens,
# interacts with truck loading, and adds a freeze-thaw term below PCI 75.
loss = (0.8
        + 0.06 * age
        + 1.2 * trucks_k * (1 + (100 - pci) / 40)
        + 0.03 * ftc * (pci < 75)
        + rng.normal(0, 1.0, n))
next_pci = np.clip(pci - np.maximum(loss, 0), 0, 100)

data = pd.DataFrame({
    "pci": pci.round(1), "aadt": aadt.round(0), "truck_share": truck.round(3),
    "age": age.round(1), "freeze_thaw": ftc, "next_pci": next_pci.round(1),
})
data["loss"] = data.pci - data.next_pci         # modelling target
print(data.head().to_string())
print(f"\nMean one-year loss: {data.loss.mean():.2f} PCI points")


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

features = ["pci", "aadt", "truck_share", "age", "freeze_thaw"]
X_tr, X_te, y_tr, y_te = train_test_split(data[features], data["loss"],
                                          test_size=0.25, random_state=42)
models = {
    "Linear regression": LinearRegression(),
    "Random forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                           random_state=42, n_jobs=-1),
    "Gradient boosting": HistGradientBoostingRegressor(max_iter=400,
                                                       learning_rate=0.05,
                                                       random_state=42),
}
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def report(name, cv_rmse, pred):
    print(f"{name:<20}{cv_rmse:>9}{mean_squared_error(y_te, pred) ** 0.5:>11.2f}"
          f"{mean_absolute_error(y_te, pred):>10.2f}{r2_score(y_te, pred):>9.2f}")

print(f"\n{'Model':<20}{'CV RMSE':>9}{'Test RMSE':>11}{'Test MAE':>10}{'Test R2':>9}")
report("Mean-loss baseline", "-", np.full(len(y_te), y_tr.mean()))
cv_scores = {}
for name, m in models.items():
    cv_scores[name] = -cross_val_score(m, X_tr, y_tr, cv=cv,
                          scoring="neg_root_mean_squared_error").mean()
    m.fit(X_tr, y_tr)
    report(name, f"{cv_scores[name]:.2f}", m.predict(X_te))

best_name = min(cv_scores, key=cv_scores.get)   # selection uses CV only
best = models[best_name]
print(f"\nSelected by cross-validation: {best_name}")

imp = permutation_importance(best, X_te, y_te, n_repeats=30, random_state=0,
                             scoring="neg_root_mean_squared_error")
print("Permutation importance (increase in test RMSE when shuffled):")
for i in np.argsort(-imp.importances_mean):
    print(f"  {features[i]:<12} {imp.importances_mean[i]:5.2f} "
          f"+/- {imp.importances_std[i]:.2f}")


THRESHOLD = 70.0
test = X_te.copy()
test["next_true"] = test.pci - y_te
test["next_pred"] = test.pci - best.predict(X_te)

acceptable = test[test.pci >= THRESHOLD]        # segments not yet below threshold
flag  = acceptable.next_pred < THRESHOLD
truth = acceptable.next_true < THRESHOLD
tp, fp = int((flag & truth).sum()), int((flag & ~truth).sum())
fn, tn = int((~flag & truth).sum()), int((~flag & ~truth).sum())
print(f"\nTest segments with PCI >= {THRESHOLD:.0f}: {len(acceptable)}")
print(f"Flagged: {int(flag.sum())} (TP={tp}, FP={fp}, FN={fn}, TN={tn}); "
      f"precision = {tp / (tp + fp):.2f}, recall = {tp / (tp + fn):.2f}")

ranked = (acceptable[flag]
          .assign(pred_loss=lambda d: d.pci - d.next_pred)
          .sort_values("pred_loss", ascending=False))
cols = ["pci", "aadt", "truck_share", "age", "next_pred", "next_true"]
print("\nFive flagged segments with the largest predicted loss:")
print(ranked[cols].head().round({"next_pred": 1, "next_true": 1}).to_string())

# Query two hypothetical segments with equal current PCI
query = pd.DataFrame([
    {"pci": 74.0, "aadt": 16_000, "truck_share": 0.20, "age": 18, "freeze_thaw": 60},
    {"pci": 74.0, "aadt": 3_000,  "truck_share": 0.05, "age": 6,  "freeze_thaw": 20},
])
query["next_pred"] = query.pci - best.predict(query[features])
print("\n", query.round(2).to_string(index=False))


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
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
order = np.argsort(imp.importances_mean)
ax[0].barh([features[i] for i in order], imp.importances_mean[order],
           xerr=imp.importances_std[order], color=GOLD, ecolor=INK)
ax[0].set(xlabel="Increase in test RMSE (PCI points)",
          title=f"(a) Permutation importance, {best_name.lower()}")

groups = [("Flagged, correct", flag & truth, GOLD, "o"),
          ("Not flagged, correct", ~flag & ~truth, SAGE, "o"),
          ("Missed or false flag", flag != truth, RUST, "X")]
for label, m, color, marker in groups:
    ax[1].scatter(acceptable.next_pred[m], acceptable.next_true[m], s=24,
                  color=color, marker=marker, label=f"{label} ({m.sum()})")
ax[1].axvline(THRESHOLD, ls="--", color=OLIVE, lw=1.2)
ax[1].axhline(THRESHOLD, ls="--", color=OLIVE, lw=1.2)
ax[1].set(xlabel="Predicted next-year PCI", ylabel="True next-year PCI",
          title=f"(b) Screening of {len(acceptable)} segments at PCI >= 70")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("chap06_fig01_pavement_screening.png", dpi=200)
