"""Chapter 2 - Case Study 2: A Graph Neural Network for Traffic Volume on Uncounted Local Roads.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/02a.tex (3 listing block(s), concatenated in book order).
Run from the repository root with `make ch02` or `python scripts/run_all.py --chapter 2`;
figures are written to outputs/ch02/.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import networkx as nx

# ---------- Synthetic road network: 12 x 12 grid, three functional classes ----------
rng = np.random.default_rng(5)
G = nx.grid_2d_graph(12, 12)
pos = {n: np.array(n, float) + rng.normal(0, 0.12, 2) for n in G.nodes}   # jittered geometry
edges = list(G.edges)
def road_class(u, v):
    """Arterials every 4th row/column, collectors every 2nd, local otherwise."""
    line = u[0] if u[0] == v[0] else u[1]
    return 2 if line % 4 == 0 else (1 if line % 2 == 0 else 0)          # 0 local, 1 collector, 2 arterial
cls = np.array([road_class(u, v) for u, v in edges])
lanes = np.where(cls == 2, 4, 2)
speed = np.where(cls == 2, 45, np.where(cls == 1, 35, 25))
pop = rng.lognormal(np.log(300), 0.6, len(edges)).round()                # population within 500 m
btw = nx.edge_betweenness_centrality(G, weight=None)
centrality = np.array([btw[e] if e in btw else btw[(e[1], e[0])] for e in edges])

# Latent "true" AADT: functional class dominates, plus population, centrality, and a
# spatially correlated random field so that neighbours share unobserved demand.
base = np.exp(np.array([6.0, 7.2, 8.6])[cls])                            # ~400 / 1300 / 5400
L = nx.line_graph(G)                                                     # segments as nodes
idx = {e: i for i, e in enumerate(edges)}
A = np.zeros((len(edges), len(edges)))
for a, b in L.edges:
    i = idx[a] if a in idx else idx[(a[1], a[0])]; j = idx[b] if b in idx else idx[(b[1], b[0])]
    A[i, j] = A[j, i] = 1
field = rng.normal(0, 1, len(edges))
for _ in range(4):                                                       # smooth the field over the graph
    field = 0.5 * field + 0.5 * (A @ field) / np.maximum(A.sum(1), 1)
field = (field - field.mean()) / field.std()
aadt = (base * (pop / 300) ** 0.35 * np.exp(0.6 * (centrality / centrality.mean() - 1) * 0.3)
        * np.exp(0.35 * field) * np.exp(rng.normal(0, 0.15, len(edges)))).round()

seg = pd.DataFrame({"cls": cls, "lanes": lanes, "speed": speed, "pop": pop,
                    "centrality": centrality, "aadt": aadt})
# Counts exist mostly on arterials and collectors: 90% / 50% / 12% coverage by class
p_count = np.array([0.12, 0.50, 0.90])[cls]
seg["counted"] = rng.random(len(edges)) < p_count
print(f"Segments: {len(seg)}; counted: {seg.counted.sum()} "
      f"({100 * seg.counted.mean():.0f}%); uncounted local roads: "
      f"{((seg.cls == 0) & ~seg.counted).sum()}")
print(seg.groupby("cls").agg(n=("aadt", "size"), counted=("counted", "sum"),
                             median_aadt=("aadt", "median")).rename(
      index={0: "local", 1: "collector", 2: "arterial"}).to_string())


# ---------- Three estimators for uncounted segments ----------
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras

y = np.log(seg.aadt.values)
attr = ["cls", "lanes", "speed", "pop", "centrality"]
X = seg[attr].copy(); X["pop"] = np.log(X["pop"])
A_hat = A + np.eye(len(A)); dg = A_hat.sum(1)
A_norm = tf.constant((A_hat / np.sqrt(np.outer(dg, dg))).astype(np.float32))  # D^-1/2 (A+I) D^-1/2

class GCNLayer(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__(); self.dense = keras.layers.Dense(units, activation=activation)
    def call(self, x, a): return self.dense(tf.matmul(a, x))          # H' = sigma(A_norm H W)
class GCN(keras.Model):
    def __init__(self):
        super().__init__(); self.g1 = GCNLayer(32, "relu"); self.g2 = GCNLayer(16, "relu")
        self.out = keras.layers.Dense(1); self.drop = keras.layers.Dropout(0.2)
    def call(self, x, a, training=False):
        h = self.drop(self.g1(x, a), training=training)
        return tf.squeeze(self.out(self.g2(h, a)), -1)

def fit_all(counted, seed=0):
    """Return log-AADT predictions for every segment from three estimators."""
    tr = counted
    cls_med = np.array([np.median(seg.aadt[tr & (seg.cls.values == c)]) for c in range(3)])[seg.cls]
    resid = y - np.log(cls_med)                                  # class-detrended log volume
    nbr_n = A @ tr.astype(float)
    nbr_resid = np.where(nbr_n > 0, (A @ (tr * resid)) / np.maximum(nbr_n, 1), np.nan)
    # (a) attributes only
    pa = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=5,
                                       random_state=seed).fit(X[tr], y[tr]).predict(X)
    # (b) attributes + neighbourhood features (mean detrended log-AADT of counted neighbours)
    Xb = X.copy(); Xb["nbr_resid_mean"] = nbr_resid; Xb["nbr_known_n"] = nbr_n
    Xb["nbr_cls_mean"] = (A @ seg.cls.values) / np.maximum(A.sum(1), 1)
    pb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=5,
                                       random_state=seed).fit(Xb[tr], y[tr]).predict(Xb)
    # (c) GCN with the same inputs; trained on counted nodes only, predicts deviation from class median
    keras.utils.set_random_seed(seed)
    F = np.column_stack([np.eye(3)[seg.cls], np.log(seg["pop"]), seg.centrality / seg.centrality.std(),
                         seg.lanes / 4, np.nan_to_num(nbr_resid, nan=0.0), nbr_n]).astype(np.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    gcn, opt = GCN(), keras.optimizers.Adam(1e-2)
    Ft, yt = tf.constant(F), tf.constant(y.astype(np.float32))
    mask, base = tf.constant(tr.astype(np.float32)), tf.constant(np.log(cls_med).astype(np.float32))
    for epoch in range(600):
        with tf.GradientTape() as tape:
            out = gcn(Ft, A_norm, training=True) + base
            loss = (tf.reduce_sum(mask * (out - yt) ** 2) / tf.reduce_sum(mask)
                    + 1e-3 * sum(tf.reduce_sum(w ** 2) for w in gcn.trainable_variables))
        opt.apply_gradients(zip(tape.gradient(loss, gcn.trainable_variables),
                                gcn.trainable_variables))
    pc = (gcn(Ft, A_norm) + base).numpy()
    return {"Class median (lookup table)": np.log(cls_med), "(a) Gradient boosting, attributes": pa,
            "(b) + neighbourhood features": pb, "(c) Graph convolutional network": pc}

def metrics(p, test):
    loc = test & (seg.cls.values == 0)
    return (mean_absolute_error(seg.aadt[test], np.exp(p[test])), r2_score(y[test], p[test]),
            mean_absolute_error(seg.aadt[loc], np.exp(p[loc])),
            100 * np.mean(np.abs(np.exp(p[loc]) - seg.aadt[loc]) / seg.aadt[loc]))

preds = fit_all(seg.counted.values)
test = ~seg.counted.values
print(f"\n{'Estimator (uncounted segments, n=%d)' % test.sum():<40}{'MAE':>8}{'R2 log':>8}"
      f"{'MAE loc':>9}{'MAPE loc':>9}")
for name, p in preds.items():
    m = metrics(p, test)
    print(f"{name:<40}{m[0]:>8.0f}{m[1]:>8.3f}{m[2]:>9.0f}{m[3]:>9.1f}")

# Robustness: redraw which segments are counted (same coverage rates) five times
scores = {k: [] for k in preds}
for s in range(1, 6):
    counted_s = np.random.default_rng(100 + s).random(len(seg)) < p_count
    for k, p in fit_all(counted_s, seed=s).items():
        scores[k].append(metrics(p, ~counted_s)[1])
print("\nR2 (log AADT) on uncounted segments over five redraws of the count locations:")
for k, v in scores.items():
    print(f"  {k:<40}{np.mean(v):.3f} +/- {np.std(v):.3f}")


# ---------- Figure: network and predictions ----------
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
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4),
                       gridspec_kw={"width_ratios": [1.15, 1]})
colors = np.array([SKY, SAGE, "#4E6A61"])     # local, collector, arterial
for (u, v), c, k in zip(edges, seg.cls, seg.counted):
    ax[0].plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=colors[c],
               lw=[1.2, 2.2, 3.4][c], alpha=1.0 if k else 0.45)
    if k:
        ax[0].plot((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2,
                   "o", ms=3, color=RUST)
ax[0].set(title="(a) Synthetic network: class by width, counts as dots",
          xticks=[], yticks=[])
ax[0].grid(False)
ax[0].set_aspect("equal")
loc = test & (seg.cls.values == 0)
for name, mk, col in [("(a) Gradient boosting, attributes", "s", SAGE),
                      ("(c) Graph convolutional network", "o", GOLD)]:
    ax[1].loglog(seg.aadt[loc], np.exp(preds[name][loc]), mk, ms=4,
                 alpha=0.8, color=col, label=name)
lims = [150, 2500]
ax[1].plot(lims, lims, "--", color=OLIVE, lw=1.2)
ax[1].set(xlim=lims, ylim=lims,
          xlabel="True AADT (uncounted local segments)",
          ylabel="Estimated AADT", title="(b) Local-road estimates")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("chap02_fig02_gnn_volume.png", dpi=200)
