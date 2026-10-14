"""GridCast core: features, direct multi-horizon models, conformal intervals, explainability."""
import numpy as np, pandas as pd, joblib, json
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from data_gen import add_calendar

H = np.arange(1, 25)                      # forecast horizons (hours ahead)
MIN_ORIGIN = 400                          # need >= 336h of history

def load_dataset(path="data/campus_load.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.asfreq("h").interpolate(limit=6)
    if "holiday" not in df or "phase" not in df: df = add_calendar(df)
    return df

def build_features(df, origins, temp_shift=0.0, temp_override=None):
    """One row per (origin, horizon). Direct strategy: no recursive error build-up.
    Weather & calendar at TARGET time are treated as known (weather forecast)."""
    L = df["load"].values
    T = df["temperature"].values + temp_shift
    if temp_override is not None: T = temp_override
    Hm = df["humidity"].values
    s = df["load"]
    r24 = s.rolling(24).mean().values; r168 = s.rolling(168).mean().values
    sd24 = s.rolling(24).std().values
    t24 = pd.Series(T).rolling(24, min_periods=1).mean().values
    p = np.asarray(origins)[:, None]; q = p + H[None, :]
    hh = df.index.hour.values; dw = df.index.dayofweek.values; mo = df.index.month.values
    P = np.broadcast_to(p, q.shape)
    f = {
     "horizon": np.broadcast_to(H, q.shape),
     "load_now": L[P], "load_prev_hr": L[P-1], "trend_24h": L[P]-L[P-24],
     "mean_24h": r24[P], "mean_7d": r168[P], "std_24h": sd24[P],
     "same_hr_yday": L[q-24], "same_hr_2d": L[q-48], "same_hr_lastwk": L[q-168],
     "same_hr_mean7d": np.mean([L[q-24*k] for k in range(1, 8)], axis=0),
     "temp": T[q], "humidity": Hm[q], "temp_mean_24h": t24[P],
     "cooling_deg": np.clip(T[q]-25, 0, None),
     "heat_index": T[q] + 0.1*np.clip(Hm[q]-60, 0, None),
     "hour": hh[q], "hour_sin": np.sin(2*np.pi*hh[q]/24), "hour_cos": np.cos(2*np.pi*hh[q]/24),
     "dayofweek": dw[q], "is_weekend": (dw[q] >= 5).astype(int), "month": mo[q],
     "holiday": df["holiday"].values[q], "holiday_yday": df["holiday"].values[q-24],
     "academic_phase": df["phase"].values[q],
    }
    X = pd.DataFrame({k: np.ravel(v) for k, v in f.items()})
    meta = pd.DataFrame({"origin_pos": np.ravel(P), "target_pos": np.ravel(q)})
    meta["target_time"] = df.index[meta.target_pos.values]
    meta["y"] = L[meta.target_pos.values] if q.max() < len(L) else np.nan
    return X, meta

def mape(y, p): y = np.asarray(y); return float(np.mean(np.abs((y-np.asarray(p))/y))*100)
def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

LGB_PARAMS = dict(n_estimators=500, learning_rate=0.04, num_leaves=48, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)

def train(df, step=2, alpha=0.2):
    n = len(df); last = n - 25
    origins = np.arange(MIN_ORIGIN, last, step)
    X, meta = build_features(df, origins)
    pos = meta.origin_pos.values
    a, b = int(n*.70), int(n*.85)                    # chronological split
    tr, va, te = pos < a, (pos >= a) & (pos < b), pos >= b
    y = meta.y.values
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(X[tr], y[tr])
    gbm = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[tr], y[tr])
    qlo = lgb.LGBMRegressor(objective="quantile", alpha=alpha/2, **LGB_PARAMS).fit(X[tr], y[tr])
    qhi = lgb.LGBMRegressor(objective="quantile", alpha=1-alpha/2, **LGB_PARAMS).fit(X[tr], y[tr])
    pr, pg = ridge.predict(X[va]), gbm.predict(X[va])
    w = min(np.linspace(0, 1, 21), key=lambda w: mape(y[va], w*pg+(1-w)*pr))   # blend weight
    # conformalised quantile regression: widen band using validation residuals
    lo, hi = qlo.predict(X[va]), qhi.predict(X[va])
    score = np.maximum(lo-y[va], y[va]-hi)
    Q = float(np.quantile(score, min(1, (1-alpha)*(1+1/len(score)))))
    bundle = dict(ridge=ridge, gbm=gbm, qlo=qlo, qhi=qhi, w=float(w), Q=Q, cols=list(X.columns), alpha=alpha)
    # ---- evaluation on untouched test period
    Xt, yt = X[te], y[te]; mt = meta[te]
    pred = predict_bundle(bundle, Xt)
    preds = {"Seasonal naive (same hr last week)": Xt.same_hr_lastwk.values,
             "Seasonal naive (same hr yesterday)": Xt.same_hr_yday.values,
             "Ridge regression": ridge.predict(Xt), "LightGBM": gbm.predict(Xt),
             "Ensemble (Ridge+LightGBM)": pred["mean"]}
    rows = [dict(model=k, MAPE=mape(yt, v), MAE=mae(yt, v), RMSE=rmse(yt, v)) for k, v in preds.items()]
    res = pd.DataFrame(rows).sort_values("MAPE")
    hor = pd.DataFrame({"horizon": Xt.horizon.values, "ape": np.abs((yt-pred["mean"])/yt)*100}).groupby("horizon").ape.mean()
    cover = float(np.mean((yt >= pred["lo"]) & (yt <= pred["hi"]))*100)
    return bundle, dict(table=res, by_horizon=hor, coverage=cover, split=(a, b), test_start=str(df.index[b]))

def predict_bundle(b, X):
    X = X[b["cols"]]
    mean = b["w"]*b["gbm"].predict(X) + (1-b["w"])*b["ridge"].predict(X)
    lo = np.minimum(b["qlo"].predict(X), mean) - b["Q"]; hi = np.maximum(b["qhi"].predict(X), mean) + b["Q"]
    return dict(mean=mean, lo=np.maximum(lo, 0), hi=hi)

def forecast(b, df, origin_pos, temp_shift=0.0):
    X, meta = build_features(df, [origin_pos], temp_shift=temp_shift)
    out = predict_bundle(b, X)
    res = pd.DataFrame({"time": meta.target_time.values, "forecast": out["mean"], "lo": out["lo"], "hi": out["hi"]})
    res["actual"] = meta.y.values
    return res, X

def save(bundle, report, path="artifacts"):
    joblib.dump(bundle, f"{path}/bundle.joblib")
    report["table"].to_csv(f"{path}/model_comparison.csv", index=False)
    report["by_horizon"].to_csv(f"{path}/mape_by_horizon.csv")
    json.dump({"coverage": report["coverage"], "test_start": report["test_start"]}, open(f"{path}/meta.json", "w"))
