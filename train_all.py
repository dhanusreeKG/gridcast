import os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import json, numpy as np, pandas as pd, shap, joblib
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from core import load_dataset, build_features, LGB_PARAMS, mape, mae, rmse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(1); np.random.seed(1)

df = load_dataset(); df = df.loc[:"2026-10-15 23:00"]          # data cutoff: 14 Oct 2026 (+ tomorrow's weather)
pos = lambda s: df.index.get_loc(pd.Timestamp(s))
A, B, T0, END = pos("2026-03-01"), pos("2026-06-24"), pos("2026-06-24"), pos("2026-10-14 23:00")
L = df.load.values

# ---------------- LightGBM + Ridge (direct 24h, origin 23:00 previous day)
org_tr = np.arange(400, A-25, 2); X, m = build_features(df, org_tr)
y = m.y.values
ridge = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(X, y)
gbm = lgb.LGBMRegressor(**{**LGB_PARAMS, **({'n_estimators':100} if os.environ.get('QUICK') else {})}).fit(X, y)
days = pd.date_range("2026-06-24", "2026-10-15"); d_pos = np.array([pos(d) for d in days])
org_d = d_pos - 1                                        # origin = 23:00 of the previous day
Xd, md = build_features(df.assign(load=df.load), org_d)
P = dict(lgbm=gbm.predict(Xd).reshape(-1,24), ridge=ridge.predict(Xd).reshape(-1,24), naive=Xd.same_hr_lastwk.values.reshape(-1,24))

# ---------------- LSTM
SEQ = 72
ph1 = np.eye(5)[df.phase.values]
hh, dw = df.index.hour.values, df.index.dayofweek.values
Fh = np.column_stack([L/1000, (df.temperature.values-28)/6, (df.humidity.values-70)/15, df.holiday.values, ph1,
                      np.sin(2*np.pi*hh/24), np.cos(2*np.pi*hh/24), np.sin(2*np.pi*dw/7), np.cos(2*np.pi*dw/7)]).astype("float32")
FUT_IDX = [1,2,3,4,5,6,7,8,9,10,11,12]                  # temp,hum,holiday,phase(5),hsin,hcos,dsin,dcos
def lstm_inputs(origins, shift=0.0):
    o = np.asarray(origins)[:, None]; q = o + np.arange(1, 25)[None, :]
    past = np.stack([Fh[p-SEQ+1:p+1] for p in origins])
    fut = Fh[q][:, :, FUT_IDX].copy(); fut[:, :, 0] += shift/6
    lw = (L[q-168]/1000).astype("float32")[..., None]; yd = (L[q-24]/1000).astype("float32")[..., None]
    return past, np.concatenate([fut, lw, yd], -1).astype("float32"), (L[q]/1000).astype("float32")
tr_o = np.arange(400, A-25, 2); va_o = np.arange(A, B-25, 1)
Xp, Xf, Y = lstm_inputs(tr_o); Vp, Vf, VY = lstm_inputs(va_o)
ip, i2 = keras.Input((SEQ, Fh.shape[1])), keras.Input((24, Xf.shape[2]))
h = layers.LSTM(64, name="lstm")(ip)
x = layers.Concatenate()([h, layers.Flatten()(i2)])
x = layers.Dense(128, activation="relu", name="d1")(x)
x = layers.Dense(24, name="d2")(x)
out = layers.Add()([x, layers.Lambda(lambda t: t[:, :, 12])(i2)])      # residual on same-hour-last-week
net = keras.Model([ip, i2], out); net.compile(keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.05))
net.fit([Xp, Xf], Y, validation_data=([Vp, Vf], VY), epochs=(3 if os.environ.get('QUICK') else 30), batch_size=128, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
dp, df_, dY = lstm_inputs(org_d)
P["lstm"] = net.predict([dp, df_], verbose=0)*1000
P["blend"] = (P["lstm"] + P["lgbm"])/2

# ---------------- metrics on days with actuals (26 Jun .. 14 Oct)
nd = len(days) - 1; act = np.stack([L[p:p+24] for p in d_pos[:nd]])
names = dict(lstm="LSTM (deep learning)", lgbm="LightGBM", blend="LSTM + LightGBM blend", ridge="Ridge regression", naive="Same hour last week")
rows = [dict(key=k, m=names[k], mape=round(mape(act, P[k][:nd]),2), mae=round(mae(act, P[k][:nd]),1), rmse=round(rmse(act, P[k][:nd]),1)) for k in names]
rows.sort(key=lambda r: r["mape"]); print(pd.DataFrame(rows).to_string(index=False))

# ---------------- SHAP (LightGBM)
Xs, _ = build_features(df, np.arange(B, END-25, 7)); Xs = Xs.sample(min(2000, len(Xs)), random_state=0)
sv = shap.TreeExplainer(gbm).shap_values(Xs)
imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False).head(10).round(1)
print(imp.to_string())

# ---------------- export for the web dashboard
W = {}
for lyr in ["lstm", "d1", "d2"]:
    ws = net.get_layer(lyr).get_weights(); W[lyr] = [np.round(w, 4).tolist() for w in ws]
a0 = pos("2026-06-17")
sub = df.iloc[a0:]
loadv = [int(round(v)) if i <= END-a0 else None for i, v in enumerate(sub.load.values)]
from calendar2026 import day_info
cal = {str(d.date()): day_info(d) for d in pd.date_range("2026-06-01", "2026-11-30")}
out_j = dict(start=str(sub.index[0]), n=len(days), first=str(days[0].date()), last_actual="2026-10-14",
  load=loadv, temp=np.round(sub.temperature.values,1).tolist(), hum=np.round(sub.humidity.values).astype(int).tolist(),
  hol=sub.holiday.tolist(), ph=sub.phase.tolist(),
  F={k: np.round(P[k]).astype(int).tolist() for k in ["lgbm","ridge","naive"]},
  net=W, metrics=rows, shap={k: float(v) for k, v in imp.items()}, cal=cal)
json.dump(out_j, open("web/data.json","w"), separators=(",",":"))
# parity reference for JS check
print("saved", os.path.getsize("web/data.json")//1024, "KB")
