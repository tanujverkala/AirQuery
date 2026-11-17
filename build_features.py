"""Build the stacked multi-horizon feature table for AirQuery.

One row = (station, origin time t, horizon h). The target is hourly PM2.5 at t+h.
Everything about the pollutants uses information up to and including t only.
Weather for t+h comes from a FORECAST issued 24*N hours before t+h, where N is the
smallest lead with 24*N >= h + L_HOURS (L_HOURS = assumed publication delay).

Inputs : data/clean/hourly_all.csv        (clean_data.py output)
         data/weather/weather_leads.csv   (fetch_weather.py output)
Output : data/features/features.parquet   (falls back to .csv.gz if pyarrow is missing)

Usage  : python build_features.py [--base PATH]
No learned preprocessing happens here (no scaling, no imputation), so there is
nothing to fit on train only. Splitting by time is done downstream on target_ts.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

# ---- settings (move to config.yaml later) ----
TRAIN_START = pd.Timestamp("2024-01-22 00:00")  # first target time with all 3 weather leads
HORIZONS = list(range(3, 49, 3))                # 3, 6, ..., 48 hours ahead
LEADS = [1, 2, 3]                               # previous_day1..3
L_HOURS = 6                                     # assumed publication delay (your choice)
LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 168]         # pm25_lag0 = value at origin time
ROLL = [6, 24]                                  # rolling means over the last 6 / 24 hours
OTHER_POLL = ["pm10", "no2", "so2", "co", "o3"]
OTHER_LAGS = [0, 24]
WX_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
REQUIRE_CURRENT_PM25 = True                     # drop rows with no PM2.5 at the origin hour
SEED = 42


def lead_for(h):
    for n in LEADS:
        if 24 * n >= h + L_HOURS:
            return n
    raise ValueError(f"no lead covers horizon {h}")


def station_rows(st, st_id, poll_s, wx_s, idx):
    p = poll_s.set_index("ts").reindex(idx)
    w = wx_s.set_index("ts").reindex(idx)

    base = pd.DataFrame(index=idx)
    for k in LAGS:
        base[f"pm25_lag{k}"] = p["pm25"].shift(k)
    for win in ROLL:
        base[f"pm25_mean{win}h"] = p["pm25"].rolling(win, min_periods=win // 2).mean()
    for c in OTHER_POLL:
        for k in OTHER_LAGS:
            base[f"{c}_lag{k}"] = p[c].shift(k)

    parts = []
    for h in HORIZONS:
        n = lead_for(h)
        k = int(np.ceil(h / 24))
        f = base.copy()
        f["station"] = st
        f["station_id"] = np.int8(st_id)
        f["horizon"] = np.int16(h)
        f["origin_ts"] = idx
        f["target_ts"] = idx + pd.Timedelta(hours=h)
        f["target_pm25"] = p["pm25"].shift(-h)
        # same hour on the most recent day that is already observed at the origin
        f["base_yesterday"] = p["pm25"].shift(24 * k - h)
        for v in WX_VARS:
            f[f"wx_{v}"] = w[f"{v}_lead{n}"].shift(-h)
        wd = np.deg2rad(w[f"wind_direction_10m_lead{n}"].shift(-h))
        f["wx_wd_sin"], f["wx_wd_cos"] = np.sin(wd), np.cos(wd)
        tt = f["target_ts"]
        f["hour_sin"] = np.sin(2 * np.pi * tt.dt.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * tt.dt.hour / 24)
        f["doy_sin"] = np.sin(2 * np.pi * tt.dt.dayofyear / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * tt.dt.dayofyear / 365.25)
        f["dow"] = tt.dt.dayofweek.astype("int8")

        keep = (f["target_ts"] >= TRAIN_START) & f["target_pm25"].notna()
        if REQUIRE_CURRENT_PM25:
            keep &= f["pm25_lag0"].notna()
        f = f[keep].copy()
        num = f.select_dtypes("float64").columns
        f[num] = f[num].astype("float32")
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def get(frame, st, ts, col):
    try:
        return frame.loc[(st, ts), col]
    except KeyError:
        return np.nan


def selfcheck(df, poll, wx, n=300):
    """Recompute sampled cells by direct timestamp lookup (no shifting) and compare."""
    rng = np.random.default_rng(SEED)
    pi = poll.set_index(["station", "ts"]).sort_index()
    wi = wx.set_index(["station", "ts"]).sort_index()
    sample = df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)]
    H = lambda x: pd.Timedelta(hours=int(x))

    def same(a, b):
        return (pd.isna(a) and pd.isna(b)) or bool(np.isclose(a, b, rtol=1e-4, atol=1e-4))

    bad = 0
    for _, r in sample.iterrows():
        st, t0, tt, h = r["station"], r["origin_ts"], r["target_ts"], int(r["horizon"])
        n_lead, k = lead_for(h), int(np.ceil(h / 24))
        assert 24 * n_lead >= h + L_HOURS, "lead not available at prediction time"
        assert tt == t0 + H(h)
        checks = {
            "target_pm25": get(pi, st, tt, "pm25"),
            "pm25_lag0": get(pi, st, t0, "pm25"),
            "pm25_lag24": get(pi, st, t0 - H(24), "pm25"),
            "pm25_lag168": get(pi, st, t0 - H(168), "pm25"),
            "no2_lag24": get(pi, st, t0 - H(24), "no2"),
            "base_yesterday": get(pi, st, tt - H(24 * k), "pm25"),
            "wx_temperature_2m": get(wi, st, tt, f"temperature_2m_lead{n_lead}"),
            "wx_precipitation": get(wi, st, tt, f"precipitation_lead{n_lead}"),
        }
        for col, expected in checks.items():
            if not same(r[col], expected):
                bad += 1
                print("MISMATCH", st, t0, h, col, r[col], expected)
    assert bad == 0, f"{bad} mismatches in self-check"
    print(f"self-check passed on {len(sample)} random rows "
          f"({len(sample) * 8} cell comparisons)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    args, _ = ap.parse_known_args()  # ignore Colab's injected "-f kernel.json"
    base = pathlib.Path(args.base)

    poll = pd.read_csv(base / "data/clean/hourly_all.csv", parse_dates=["ts"])
    wx = pd.read_csv(base / "data/weather/weather_leads.csv", parse_dates=["ts"])
    idx = pd.date_range(poll["ts"].min(), poll["ts"].max(), freq="h")
    stations = sorted(poll["station"].unique())

    frames = []
    for i, st in enumerate(stations):
        assert (wx["station"] == st).any(), f"no weather rows for station name: {st}"
        frames.append(station_rows(st, i, poll[poll.station == st],
                                   wx[wx.station == st], idx))
        print("built:", st)
    df = pd.concat(frames, ignore_index=True)

    print("\nlead used per horizon:",
          {h: lead_for(h) for h in HORIZONS})
    print("rows:", len(df), "| target_ts from", df["target_ts"].min(),
          "to", df["target_ts"].max())
    print("\nrows per station:")
    print(df.groupby("station").size().to_string())
    print("\nrows per horizon:")
    print(df.groupby("horizon").size().to_string())
    feat = [c for c in df.columns
            if c not in ("station", "origin_ts", "target_ts", "target_pm25")]
    print("\nmissing % per feature (top 10):")
    print((df[feat].isna().mean() * 100).round(1).sort_values(ascending=False)
          .head(10).to_string())

    selfcheck(df, poll, wx)

    out = base / "data/features"
    out.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out / "features.parquet", index=False)
        print("\nsaved", out / "features.parquet")
    except ImportError:
        df.to_csv(out / "features.csv.gz", index=False)
        print("\npyarrow missing; saved", out / "features.csv.gz")


if __name__ == "__main__":
    main()
