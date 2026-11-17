"""Clean the raw CPCB station CSVs: mask invalid values, resample 15-min to hourly,
and write data/clean/hourly_all.csv (all hours kept, gaps visible). Prints a per-station
coverage report."""
import pathlib

import numpy as np
import pandas as pd

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/clean")
OUT.mkdir(parents=True, exist_ok=True)

SHIFT_HOURS = 0  # leave 0 unless you confirm the +0000 stamps are real UTC (then 5.5)

COLS = {
    "PM2.5 (µg/m³)": "pm25", "PM10 (µg/m³)": "pm10", "NO2 (µg/m³)": "no2",
    "SO2 (µg/m³)": "so2", "CO (mg/m³)": "co", "Ozone (µg/m³)": "o3",
    "RH (%)": "rh", "WS (m/s)": "ws", "WD (deg)": "wd",
}
NUM = list(COLS.values())
CORE = [c for c in NUM if c != "wd"] + ["wd_sin", "wd_cos"]

frames, report = [], []
for f in sorted(RAW.glob("*.csv")):
    df = pd.read_csv(f)
    station = df["Station Name"].iloc[0]
    df["ts"] = (pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
                + pd.Timedelta(hours=SHIFT_HOURS))
    df = df.rename(columns=COLS)[["ts"] + NUM]
    df[NUM] = df[NUM].apply(pd.to_numeric, errors="coerce")
    df[NUM] = df[NUM].mask((df[NUM] < 0) | (df[NUM] == 999))  # invalid sentinels

    rad = np.deg2rad(df["wd"])  # average wind direction as sin/cos, not raw degrees
    df["wd_sin"], df["wd_cos"] = np.sin(rad), np.cos(rad)

    h = df.drop(columns="wd").set_index("ts").resample("1h").mean()
    h["station"] = station
    frames.append(h.reset_index())

    isna = h["pm25"].isna()
    report.append({
        "station": station,
        "hours_in_span": len(h),
        "pm25_valid_hours": int((~isna).sum()),
        "pm25_coverage_%": round(100 * (~isna).mean(), 1),
        "complete_case_hours": int(h[CORE].dropna().shape[0]),
        "longest_pm25_gap_h": int(isna.groupby((~isna).cumsum()).sum().max()),
    })

pd.concat(frames).to_csv(OUT / "hourly_all.csv", index=False)  # all hours kept, gaps visible
print(pd.DataFrame(report).to_string(index=False))
