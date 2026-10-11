"""Chapter 3 - Case Study 1: Short-Term Traffic Volume Forecasting.

Book: Artificial Intelligence in Highway Engineering
Source: chapters/03a.tex (3 listing block(s), concatenated in book order).
Run from the repository root with `make ch03` or `python scripts/run_all.py --chapter 3`;
figures are written to outputs/ch03/.
"""
import numpy as np
import pandas as pd

# ---------- Synthetic hourly traffic volume series (two years) ----------
rng = np.random.default_rng(3)
idx = pd.date_range("2024-01-01", periods=24 * 731, freq="h")   # 2024-2025 (731 days)
h = idx.hour.values; dow = idx.dayofweek.values; doy = idx.dayofyear.values

def bimodal(hour):          # weekday profile: AM and PM peaks
    return (np.exp(-((hour - 7.5) / 1.6) ** 2) + 1.2 * np.exp(-((hour - 17) / 1.9) ** 2)
            + 0.25 * np.exp(-((hour - 12.5) / 3.0) ** 2))
def unimodal(hour):         # weekend profile: single midday plateau
    return 0.9 * np.exp(-((hour - 13.5) / 3.5) ** 2)

weekday = dow < 5
base = 900 + 4300 * np.where(weekday, bimodal(h), unimodal(h))
base *= 1 + 0.06 * np.sin(2 * np.pi * (doy - 200) / 365)      # summer high, winter low
base *= 1 + 0.02 * (idx.year.values - 2024)                    # slow growth
holidays = pd.to_datetime(["2024-01-01", "2024-05-27", "2024-07-04", "2024-09-02",
                           "2024-11-28", "2024-12-25", "2025-01-01", "2025-05-26",
                           "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"])
is_holiday = np.isin(idx.normalize(), holidays)
base = np.where(is_holiday, 900 + 4300 * 0.6 * unimodal(h), base)
rain = rng.random(len(idx)) < 0.08
rain = rain | np.roll(rain, 1) | np.roll(rain, 2)               # rain lasts a few hours
base *= np.where(rain, 0.93, 1.0)
volume = np.clip(base + rng.normal(0, 180, len(idx)), 0, None).round()

df = pd.DataFrame({"traffic_volume": volume, "rain": rain.astype(int),
                   "holiday": is_holiday.astype(int)}, index=idx)
print(df.describe().round(1).T.to_string())
print(f"\nMean weekday volume {df.traffic_volume[weekday].mean():.0f}; "
      f"weekend {df.traffic_volume[~weekday].mean():.0f}; "
      f"holiday {df.traffic_volume[is_holiday == 1].mean():.0f}")
df.to_csv("synthetic_hourly_volume.csv")


# ---------- Forecasting models and rolling evaluation ----------
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sklearn.ensemble import HistGradientBoostingRegressor
import statsmodels.api as sm

TEST_H = 24 * 56                                  # final eight weeks held out
y = df.traffic_volume
train, test = y.iloc[:-TEST_H], y.iloc[-TEST_H:]

def scores(actual, pred):
    e = np.asarray(actual) - np.asarray(pred)
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))

results = {}

# 1. Seasonal naive: same hour one week earlier (valid for any horizon <= 168 h)
naive = y.shift(168).iloc[-TEST_H:]
results["Seasonal naive (168 h)"] = {"1 h": scores(test, naive), "24 h": scores(test, naive)}

# 2. SARIMA fitted once on the last 12 weeks of training, then applied without refitting
sarima = sm.tsa.SARIMAX(train.iloc[-24 * 84:], order=(2, 0, 1),
                        seasonal_order=(1, 1, 1, 24)).fit(disp=False)
one_step = sarima.apply(y.iloc[-24 * 84 - TEST_H:]).get_prediction().predicted_mean.iloc[-TEST_H:]
day_ahead = []
for d in range(56):                               # 24-step forecast issued at midnight
    end = len(y) - TEST_H + 24 * d
    res = sarima.apply(y.iloc[end - 24 * 84:end])
    day_ahead.append(res.forecast(24).values)
results["SARIMA (2,0,1)(1,1,1)24"] = {"1 h": scores(test, one_step),
                                       "24 h": scores(test, np.concatenate(day_ahead))}

# 3. Gradient boosting with calendar, weather, and lag features (direct models)
def features(min_lag):
    F = pd.DataFrame({"hs": np.sin(2 * np.pi * y.index.hour / 24),
                      "hc": np.cos(2 * np.pi * y.index.hour / 24),
                      "dow": y.index.dayofweek, "holiday": df.holiday, "rain": df.rain},
                     index=y.index)
    for L in [1, 2, 3, 24, 25, 48, 168]:
        if L >= min_lag:
            F[f"lag{L}"] = y.shift(L)
    return F.dropna()
for label, min_lag in [("1 h", 1), ("24 h", 24)]:
    F = features(min_lag); yy = y.loc[F.index]
    Ftr, Fte = F.iloc[:-TEST_H], F.iloc[-TEST_H:]
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                        random_state=0).fit(Ftr, yy.loc[Ftr.index])
    results.setdefault("Gradient boosting", {})[label] = scores(test, gbm.predict(Fte))
    if label == "24 h":
        gbm_pred = pd.Series(gbm.predict(Fte), index=test.index)

# 4. LSTM on a 168-hour window of scaled volume; recursive 24-step forecasts
from tensorflow import keras
from tensorflow.keras import layers
keras.utils.set_random_seed(0)
LOOK = 168
mu, sd = train.mean(), train.std()
z = ((y - mu) / sd).values
X = np.lib.stride_tricks.sliding_window_view(z[:-1], LOOK)[:, :, None]
t = z[LOOK:]
Xtr, ttr = X[:-TEST_H], t[:-TEST_H]
lstm = keras.Sequential([layers.Input((LOOK, 1)), layers.LSTM(32), layers.Dense(1)])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(Xtr, ttr, epochs=6, batch_size=64, validation_split=0.1, verbose=0)
p1 = lstm.predict(X[-TEST_H:], verbose=0).ravel() * sd + mu           # one step ahead
rec = []
for d in range(56):
    end = len(z) - TEST_H + 24 * d
    win = z[end - LOOK:end].copy()
    for _ in range(24):
        nxt = lstm.predict(win[None, :, None], verbose=0)[0, 0]
        rec.append(nxt); win = np.append(win[1:], nxt)
lstm_pred = pd.Series(np.array(rec) * sd + mu, index=test.index)
results["LSTM (168-h window)"] = {"1 h": scores(test, p1), "24 h": scores(test, lstm_pred)}

print(f"{'Model':<26}{'MAE 1h':>8}{'RMSE 1h':>9}{'MAE 24h':>9}{'RMSE 24h':>10}")
for m, r in results.items():
    print(f"{m:<26}{r['1 h'][0]:>8.0f}{r['1 h'][1]:>9.0f}{r['24 h'][0]:>9.0f}{r['24 h'][1]:>10.0f}")
hol = test.index[df.holiday.iloc[-TEST_H:].values == 1]
print(f"\nHoliday hours in test: {len(hol)}; MAE (24 h) on holiday hours: "
      f"naive {scores(test[hol], naive[hol])[0]:.0f}, GBM {scores(test[hol], gbm_pred[hol])[0]:.0f}, "
      f"LSTM {scores(test[hol], lstm_pred[hol])[0]:.0f}")


# ---------- Figure: a holiday week and error summary ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
week = slice("2025-11-24", "2025-11-30")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
ax[0].plot(test[week], "k-", lw=1.2, label="Actual")
ax[0].plot(naive[week], ":", label="Seasonal naive")
ax[0].plot(gbm_pred[week], "-", label="Gradient boosting (24 h)")
ax[0].plot(lstm_pred[week], "--", label="LSTM (24 h recursive)")
ax[0].axvspan(pd.Timestamp("2025-11-27"), pd.Timestamp("2025-11-28"), color="orange", alpha=0.15)
ax[0].set(title="(a) Thanksgiving week 2025 (shaded: holiday)", ylabel="Vehicles per hour")
ax[0].legend(fontsize=7.5, ncol=2)
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
names = list(results); x = np.arange(len(names))
ax[1].bar(x - 0.18, [results[m]["1 h"][0] for m in names], 0.36, label="1 h ahead")
ax[1].bar(x + 0.18, [results[m]["24 h"][0] for m in names], 0.36, label="24 h ahead")
ax[1].axhline(np.mean(np.abs(rng.normal(0, 180, 100000))), color="k", ls="--", lw=0.8,
              label="noise floor")
ax[1].set_xticks(x); ax[1].set_xticklabels(["Naive", "SARIMA", "GBM", "LSTM"], fontsize=8)
ax[1].set(title="(b) Mean absolute error, 8-week test", ylabel="MAE (veh/h)")
ax[1].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig("chap03_fig01_traffic_forecast.png", dpi=200)
