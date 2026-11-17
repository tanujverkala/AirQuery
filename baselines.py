"""Rolling-origin folds and baselines for AirQuery (no model yet).

Evaluated on the VALIDATION windows only:
  persistence : PM2.5 at the origin hour
  yesterday   : same hour as the target on the most recent day already observed
  climatology : mean PM2.5 by (station, month, hour), fit on the fold's TRAINING rows
                only; falls back to (station, hour) for months absent from training
The held-out test window is never touched unless you pass --final-test (use once, at the end).

Other scripts can import: FOLDS, FINAL, split, mae, rmse.
Usage: python baselines.py [--base PATH] [--final-test]
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

# (train_start, eval_start, eval_end). Training rows have target_ts in [train_start, eval_start),
# evaluation rows have target_ts in [eval_start, eval_end).
FOLDS = {
    "fold1": ("2024-01-22", "2024-09-01", "2025-01-01"),  # train Jan-Aug 2024, validate Sep-Dec 2024
    "fold2": ("2024-01-22", "2025-01-01", "2025-04-01"),  # train 2024, validate Jan-Mar 2025
    "fold3": ("2024-01-22", "2025-07-01", "2025-10-01"),  # train to Jun 2025, validate Jul-Sep 2025
}
FINAL = ("2024-01-22", "2025-10-01", "2026-01-01")        # train to Sep 2025, test Oct-Dec 2025
SHOW = [3, 12, 24, 36, 48]                                # horizons printed in the summary


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def split(df, spec):
    start, cut, end = (pd.Timestamp(x) for x in spec)
    t = df["target_ts"]
    tr = (t >= start) & (t < cut)
    ev = (t >= cut) & (t < end)
    assert t[tr].max() < t[ev].min(), "training targets must precede evaluation targets"
    return tr, ev


def climatology(train, ev):
    obs = train.drop_duplicates(["station", "target_ts"])
    tt = obs["target_ts"]
    obs = obs.assign(month=tt.dt.month, hour=tt.dt.hour)
    by_mh = obs.groupby(["station", "month", "hour"])["target_pm25"].mean().rename("c_mh")
    by_h = obs.groupby(["station", "hour"])["target_pm25"].mean().rename("c_h")
    et = ev["target_ts"]
    keys = pd.DataFrame({"station": ev["station"], "month": et.dt.month, "hour": et.dt.hour})
    a = keys.join(by_mh, on=["station", "month", "hour"])["c_mh"]
    b = keys.join(by_h, on=["station", "hour"])["c_h"]
    return a.fillna(b), (a.isna() & b.notna())


def run(df, name, spec):
    tr, ev = split(df, spec)
    train, ev = df[tr], df[ev].copy()
    ev["climatology"], fallback = climatology(train, ev)
    ev["persistence"] = ev["pm25_lag0"]
    ev["yesterday"] = ev["base_yesterday"]
    names = ["persistence", "yesterday", "climatology"]
    common = ev[names].notna().all(axis=1)
    info = dict(fold=name, train_rows=len(train), eval_rows=len(ev), common_rows=int(common.sum()),
                mean_pm25=float(ev["target_pm25"].mean()),
                climatology_fallback_pct=round(100 * float(fallback[common].mean()), 1))
    ev = ev[common]
    recs = []
    for h, g in ev.groupby("horizon"):
        m0 = mae(g["target_pm25"], g["persistence"])
        for b in names:
            m = mae(g["target_pm25"], g[b])
            recs.append(dict(fold=name, horizon=int(h), baseline=b, n=len(g), mae=m,
                             rmse=rmse(g["target_pm25"], g[b]), skill_vs_persistence=1 - m / m0))
    return info, pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--final-test", action="store_true")
    args, _ = ap.parse_known_args()  # ignore Colab's injected "-f kernel.json"
    base = pathlib.Path(args.base)
    path = base / "data/features/features.parquet"
    df = pd.read_parquet(path) if path.exists() else pd.read_csv(
        base / "data/features/features.csv.gz", parse_dates=["origin_ts", "target_ts"])

    if args.final_test:
        print("WARNING: evaluating the held-out TEST window. Use this once, after all tuning.")
        todo, tag = {"test": FINAL}, "test"
    else:
        todo, tag = FOLDS, "val"

    infos, frames = [], []
    for name, spec in todo.items():
        info, res = run(df, name, spec)
        infos.append(info)
        frames.append(res)
        print(f"\n=== {name} | train rows {info['train_rows']} | eval rows {info['eval_rows']} "
              f"(common to all baselines: {info['common_rows']}) | mean PM2.5 {info['mean_pm25']:.1f} "
              f"| climatology month-fallback {info['climatology_fallback_pct']}% ===")
        show = [h for h in SHOW if h in set(res["horizon"])]
        print("MAE (ug/m3) by horizon:")
        print(res.pivot(index="baseline", columns="horizon", values="mae")[show].round(1).to_string())
        print("Skill vs persistence (1 - MAE/MAE_persistence):")
        print(res.pivot(index="baseline", columns="horizon",
                        values="skill_vs_persistence")[show].round(2).to_string())

    out = base / "results"
    out.mkdir(exist_ok=True)
    pd.concat(frames).to_csv(out / f"baselines_{tag}.csv", index=False)
    pd.DataFrame(infos).to_csv(out / f"baselines_{tag}_info.csv", index=False)
    print("\nsaved", out / f"baselines_{tag}.csv")


if __name__ == "__main__":
    main()
