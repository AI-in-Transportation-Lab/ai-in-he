"""Chapter 2 - Case Study 1: A Neural Network for Hourly Traffic Volume.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/02a.tex (2 listing block(s), concatenated in book order).
Run from the repository root with `make ch02` or `python scripts/run_all.py --chapter 2`;
figures are written to outputs/ch02/.
"""
import numpy as np
import pandas as pd

# ---------- Synthetic hourly volumes with calendar and weather covariates (one year) ----------
rng = np.random.default_rng(11)
idx = pd.date_range("2025-01-01", periods=24 * 365, freq="h")
h, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values
weekday = dow < 5
profile = np.where(weekday,
                   np.exp(-((h - 7.5) / 1.6) ** 2) + 1.2 * np.exp(-((h - 17) / 1.9) ** 2)
                   + 0.25 * np.exp(-((h - 12.5) / 3) ** 2),
                   0.9 * np.exp(-((h - 13.5) / 3.5) ** 2))
holiday = np.isin(idx.normalize(), pd.to_datetime(
    ["2025-01-01", "2025-05-26", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"]))
temp = 12 + 14 * np.sin(2 * np.pi * (doy - 110) / 365) + 5 * np.sin(2 * np.pi * (h - 9) / 24) \
       + rng.normal(0, 2, len(idx))                                   # deg C
rain = (rng.random(len(idx)) < 0.09).astype(int)
rain = np.maximum(rain, np.roll(rain, 1))
volume = 900 + 4300 * np.where(holiday, 0.55 * 0.9 * np.exp(-((h - 13.5) / 3.5) ** 2), profile)
volume *= 1 + 0.04 * np.sin(2 * np.pi * (doy - 200) / 365)            # mild seasonality
volume *= np.where(rain, 0.92, 1.0) * (1 - 0.004 * np.clip(-temp, 0, None))  # cold snaps
volume = np.clip(volume + rng.normal(0, 170, len(idx)), 0, None).round()

df = pd.DataFrame({"volume": volume, "hour": h, "dow": dow, "holiday": holiday.astype(int),
                   "rain": rain, "temp": temp.round(1)}, index=idx)
df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
df["weekend"] = (~weekday).astype(int)
print(df[["volume", "temp", "rain", "holiday"]].describe().round(1).T.to_string())

# ---------- Temporal split and three supervised models ----------
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

TEST_H = 24 * 56                                         # last eight weeks held out
feat = ["hour_sin", "hour_cos", "weekend", "holiday", "rain", "temp"]
Xtr, Xte = df[feat].iloc[:-TEST_H], df[feat].iloc[-TEST_H:]
ytr, yte = df.volume.iloc[:-TEST_H], df.volume.iloc[-TEST_H:]
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

# Linear baseline with one-hot hour x weekend interactions (what a statistician would fit)
def onehot_design(d):
    H = pd.get_dummies(d.hour, prefix="h").astype(float)
    W = pd.concat([H, H.mul(d.weekend.values, axis=0).add_suffix("_we")], axis=1)
    return pd.concat([W, d[["holiday", "rain", "temp"]]], axis=1)
lin = Ridge(alpha=1.0).fit(onehot_design(df.iloc[:-TEST_H]), ytr)
pred_lin = lin.predict(onehot_design(df.iloc[-TEST_H:]))

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", learning_rate_init=1e-3,
                   max_iter=300, early_stopping=True, validation_fraction=0.1,
                   n_iter_no_change=15, random_state=0).fit(Xtr_s, ytr)
pred_mlp = mlp.predict(Xte_s)

gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
pred_gbm = gbm.predict(Xte)

print(f"\n{'Model':<34}{'MAE':>7}{'R2':>7}")
for name, p in [("Weekly seasonal naive", df.volume.shift(168).iloc[-TEST_H:]),
                ("Ridge, one-hot hour x weekend", pred_lin),
                ("MLP (64, 32), 6 raw features", pred_mlp),
                ("Gradient boosting, 6 raw features", pred_gbm)]:
    print(f"{name:<34}{mean_absolute_error(yte, p):>7.0f}{r2_score(yte, p):>7.3f}")
print(f"\nMLP stopped after {mlp.n_iter_} epochs; best validation R2 = {mlp.best_validation_score_:.3f}")


# ---------- What did the network learn? ----------
# Partial-dependence profiles and permutation importance
from sklearn.inspection import permutation_importance
grid = pd.DataFrame({"hour": np.arange(24)})
prof = {}
for label, we in [("weekday", 0), ("weekend", 1)]:
    g = pd.DataFrame({"hour_sin": np.sin(2 * np.pi * grid.hour / 24),
                      "hour_cos": np.cos(2 * np.pi * grid.hour / 24),
                      "weekend": we, "holiday": 0, "rain": 0, "temp": 15.0})
    prof[label] = mlp.predict(scaler.transform(g[feat]))
imp = permutation_importance(mlp, Xte_s, yte, n_repeats=10, random_state=0,
                             scoring="neg_mean_absolute_error")
print("\nPermutation importance of the MLP (increase in test MAE, veh/h):")
for i in np.argsort(-imp.importances_mean):
    m, s = imp.importances_mean[i], imp.importances_std[i]
    print(f"  {feat[i]:<9}{m:7.0f} +/- {s:.0f}")

# ---------- Figure ----------
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
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(mlp.loss_curve_, color=GOLD, lw=1.8,
           label="training loss (squared error / 2)")
ax[0].set(xlabel="Epoch", ylabel="Loss", title="(a) MLP training curve",
          yscale="log")
ax2 = ax[0].twinx()
ax2.plot(mlp.validation_scores_, color=SAGE, lw=1.8, label="validation $R^2$")
ax2.set(ylabel="Validation $R^2$", ylim=(0.8, 1.0))
ax2.grid(False); ax2.spines["right"].set_visible(True)
ax[0].legend(loc="upper right", fontsize=7, frameon=False)
ax2.legend(loc="center right", fontsize=7, frameon=False)
true_wd = 900 + 4300 * (np.exp(-((np.arange(24) - 7.5) / 1.6) ** 2)
                        + 1.2 * np.exp(-((np.arange(24) - 17) / 1.9) ** 2)
                        + 0.25 * np.exp(-((np.arange(24) - 12.5) / 3) ** 2))
true_we = 900 + 4300 * 0.9 * np.exp(-((np.arange(24) - 13.5) / 3.5) ** 2)
ax[1].plot(true_wd, "-", color=SAGE, lw=1.4, label="generating function")
ax[1].plot(prof["weekday"], "o", color=GOLD, ms=4, label="MLP, weekday")
ax[1].plot(true_we, "--", color=SAGE, lw=1.4)
ax[1].plot(prof["weekend"], "s", color=RUST, ms=4, label="MLP, weekend")
ax[1].set(xlabel="Hour of day", ylabel="Vehicles per hour",
          title="(b) Learned hour-of-day profiles (dry, 15 C)")
ax[1].legend(fontsize=7, frameon=False)
order = np.argsort(-imp.importances_mean)
ax[2].barh([feat[i] for i in order][::-1], imp.importances_mean[order][::-1],
           xerr=imp.importances_std[order][::-1], color=GOLD, ecolor=INK)
ax[2].set(xlabel="Increase in test MAE (veh/h)",
          title="(c) Permutation importance")
fig.tight_layout(); fig.savefig("chap02_fig01_mlp_traffic.png", dpi=200)
