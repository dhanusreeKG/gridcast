import numpy as np, pandas as pd
from calendar2026 import calendar_frame
def add_calendar(df):
    c = calendar_frame(df.index); df["holiday"] = c.holiday.values; df["phase"] = c.phase.values; return df
def generate(start="2023-01-01", end="2026-11-30 23:00", seed=7):
    rng = np.random.default_rng(seed); idx = pd.date_range(start, end, freq="h")
    n = len(idx); h = idx.hour.values; doy = idx.dayofyear.values
    seas = 28.5 + 3.5*np.sin(2*np.pi*(doy-100)/365); diur = 5.0*np.sin(2*np.pi*(h-9)/24)
    front = pd.Series(rng.normal(0, .35, n)).ewm(span=72).mean().values*6
    temp = seas + diur + front; hum = np.clip(75 - 2.2*(temp-28) + rng.normal(0, 4, n), 25, 98)
    df = add_calendar(pd.DataFrame({"temperature": temp, "humidity": hum}, index=idx)); df.index.name = "timestamp"
    dow = idx.dayofweek.values
    prof = np.interp(h, [0,5,7,9,12,13,15,17,19,22,24], [.38,.36,.55,.95,1,.9,1,.8,.55,.45,.38])
    occ = np.select([df.phase==0, df.phase==2, df.phase==3, df.phase==4], [.45,.92,.8,.78], 1.0)
    occ = occ*np.where(df.holiday==1, .55, 1)
    base = 260 + 640*prof*occ; cool = 38*np.clip(temp-25, 0, None)*(0.35+0.65*prof*occ)
    humid = 1.2*np.clip(hum-70, 0, None)*prof*occ
    noise = pd.Series(rng.normal(0, 1, n)).ewm(span=6).mean().values*55
    outage = rng.random(n) < .0008
    df["load"] = (base+cool+humid+noise)*np.where(outage, rng.uniform(.5,.75), 1)
    return df.round(2)
if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True); d = generate(); d.to_csv("data/campus_load.csv"); print(d.shape, d.index[-1])
