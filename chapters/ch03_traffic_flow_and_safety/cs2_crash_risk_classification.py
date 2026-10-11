"""Chapter 3 - Case Study 2: Real-Time Crash Risk Classification on Tabular Data.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/03a.tex (3 listing block(s), concatenated in book order).
Run from the repository root with `make ch03` or `python scripts/run_all.py --chapter 3`;
figures are written to outputs/ch03/.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd

# ---------- Synthetic 5-minute interval data for 40 freeway segments ----------
rng = np.random.default_rng(7)
N_SEG, N_INT = 40, 6000                        # 40 segments x 6000 intervals (~21 days)
n = N_SEG * N_INT
seg = np.repeat(np.arange(N_SEG), N_INT)
t = np.tile(np.arange(N_INT), N_SEG)
hour = (t * 5 // 60) % 24
night = ((hour >= 21) | (hour <= 5)).astype(int)
peak = ((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18))
volume = rng.normal(np.where(peak, 1500, 800) * (1 - 0.5 * night), 200).clip(50)
speed = rng.normal(np.where(peak, 78, 100) - 0.004 * (volume - 800), 6).clip(20)
speed_sd = rng.gamma(2.0, np.where(peak, 3.0, 1.6))          # speed variation (km/h)
rain = (rng.random(n) < 0.10).astype(int)
speed_sd = speed_sd * (1 + 0.3 * rain)
seg_effect = rng.normal(0, 0.4, N_SEG)[seg]                   # unobserved segment risk

def z(x): return (x - x.mean()) / x.std()
logit = (-5.9 + 0.9 * z(speed_sd) + 0.5 * z(volume) - 0.4 * z(speed)
         + 0.6 * rain + 0.3 * night + 0.4 * z(speed_sd) * rain + seg_effect)
crash = rng.binomial(1, 1 / (1 + np.exp(-logit)))

data = pd.DataFrame({"segment": seg, "t": t, "hour": hour, "night": night,
                     "volume": volume.round(), "speed": speed.round(1),
                     "speed_sd": speed_sd.round(2), "rain": rain, "crash": crash})
print(f"Intervals: {n:,}; crashes: {crash.sum():,} ({100 * crash.mean():.2f}%)")
print(data.groupby("rain").crash.mean().rename("crash rate by rain").round(4).to_string())


# ---------- Temporal split, three models, imbalance-aware evaluation ----------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance

feat = ["volume", "speed", "speed_sd", "rain", "night", "hs", "hc"]
data["hs"], data["hc"] = np.sin(2 * np.pi * data.hour / 24), np.cos(2 * np.pi * data.hour / 24)
train = data[data.t < 0.8 * N_INT]; test = data[data.t >= 0.8 * N_INT]   # last ~4 days held out
Xtr, ytr, Xte, yte = train[feat], train.crash, test[feat], test.crash
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
print(f"Train {len(train):,} / test {len(test):,}; positive-class weight {pos_weight:.1f}")

models = {}
models["Logistic regression"] = LogisticRegression(class_weight="balanced", max_iter=1000
                                                   ).fit(Xtr_s, ytr).predict_proba(Xte_s)[:, 1]
gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                     random_state=0).fit(Xtr, ytr)
models["Gradient boosting"] = gbm.predict_proba(Xte)[:, 1]

from tensorflow import keras
from tensorflow.keras import layers
keras.utils.set_random_seed(0)
mlp = keras.Sequential([layers.Input((len(feat),)), layers.Dense(32, activation="relu"),
                        layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")])
mlp.compile(optimizer="adam", loss="binary_crossentropy")
mlp.fit(Xtr_s, ytr, epochs=15, batch_size=256, validation_split=0.1, verbose=0,
        class_weight={0: 1.0, 1: pos_weight},
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
models["MLP (2 hidden layers)"] = mlp.predict(Xte_s, verbose=0).ravel()

def recall_at_alert_rate(y, p, rate):
    """Recall when the top `rate` fraction of intervals is flagged."""
    k = int(rate * len(p)); flagged = np.argsort(-p)[:k]
    return y.values[flagged].sum() / y.sum()

def prior_corrected(p, w):
    """Undo class weighting: p_corr = p / (p + (1-p)*w)."""
    return p / (p + (1 - p) * w)

print(f"\n{'Model':<24}{'ROC AUC':>9}{'PR AUC':>8}{'Rec@5%':>8}{'Rec@10%':>9}{'Brier*':>8}")
for m, p in models.items():
    pc = prior_corrected(p, pos_weight)
    print(f"{m:<24}{roc_auc_score(yte, p):>9.3f}{average_precision_score(yte, p):>8.3f}"
          f"{recall_at_alert_rate(yte, p, 0.05):>8.2f}{recall_at_alert_rate(yte, p, 0.10):>9.2f}"
          f"{brier_score_loss(yte, pc):>8.4f}")
print(f"{'Always predict base rate':<24}{0.5:>9.3f}{yte.mean():>8.3f}{0.05:>8.2f}{0.10:>9.2f}"
      f"{brier_score_loss(yte, np.full(len(yte), ytr.mean())):>8.4f}")
print("* Brier score after prior correction of the class-weighted probabilities")

# Alert-budget threshold for the gradient boosting model
p = models["Gradient boosting"]; k = int(0.05 * len(p)); thr = np.sort(p)[-k]
flag = p >= thr
tp = int((flag & (yte == 1)).sum()); fp = int((flag & (yte == 0)).sum())
print(f"\nGBM at a 5% alert budget: threshold {thr:.3f}, alerts {flag.sum():,}")
print(f"  crashes caught {tp} of {int(yte.sum())}, precision {tp / (tp + fp):.3f}, "
      f"lift {tp / (tp + fp) / yte.mean():.1f}x")

imp = permutation_importance(gbm, Xte, yte, scoring="average_precision",
                             n_repeats=10, random_state=0)
print("\nPermutation importance (drop in PR AUC):")
for i in np.argsort(-imp.importances_mean):
    print(f"  {feat[i]:<9}{imp.importances_mean[i]:6.3f} +/- {imp.importances_std[i]:.3f}")


# ---------- Figure: recall versus alert budget and calibration ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
GOLD, SAGE, SKY = "#DAB400", "#78978D", "#6ADBF1"  # Moonrise Kingdom palette
RUST, OLIVE = "#CE6B08", "#A59D30"
INK, PAPER, GRID = "#3F4A45", "#FBF8F4", "#D5D5D3" # text, background, grid
plt.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SAGE,
    "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
colors = {"Logistic regression": SAGE, "Gradient boosting": GOLD}
rates = np.linspace(0.005, 0.25, 50)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for m, p in models.items():
    ax[0].plot(100 * rates, [recall_at_alert_rate(yte, p, r) for r in rates],
               color=colors.get(m, SKY), lw=1.8, label=m)
ax[0].plot([0, 25], [0, 0.25], ":", color=RUST, lw=1.4, label="Random alerts")
ax[0].set(xlabel="Intervals flagged (%)",
          ylabel="Share of crashes preceded by an alert",
          title="(a) Recall versus alert budget")
ax[0].legend(fontsize=8, frameon=False)
for m, p in models.items():
    obs, pred = calibration_curve(yte, prior_corrected(p, pos_weight),
                                  n_bins=10, strategy="quantile")
    ax[1].plot(100 * pred, 100 * obs, "o-", ms=4, lw=1.8,
               color=colors.get(m, SKY), label=m)
ax[1].plot([0, 12], [0, 12], ":", color=RUST, lw=1.4,
           label="Perfect calibration")
ax[1].set(xlabel="Predicted crash probability (%)",
          ylabel="Observed crash rate (%)",
          title="(b) Reliability after prior correction")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("chap03_fig02_crash_risk.png", dpi=200)
