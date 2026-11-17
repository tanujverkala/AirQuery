"""Rolling (adaptive) interval calibration for AirQuery.

Same correction as calibrate.py (width-normalised, one correction per tail and horizon), but
it is re-learned every day from the most recent forecasts whose outcome is already known:

  at the start of day D (00:00), for horizon h, use the forecasts with target_ts in
  (D - window, D]  (all stations pooled), i.e. only outcomes observable before day D begins.
  Q_lo, Q_hi = conformal (1 - TAIL) quantiles of s_lo, s_hi over that window, applied to every
  forecast issued on day D. Fewer than MIN_N resolved forecasts in the window means no
  correction yet (warm-up), which is reported as 'calibrated_share'.

The scores come from the validation predictions, which are all out-of-sample for their own
model. Folds are processed on one timeline, so a window can reach back into the previous fold
(the previous model's recent forecasts) and has a warm-up after the Apr-Jun 2025 gap.
With --test, the test predictions join the same timeline and only the test rows are scored,
using outcomes available before each day. Use --test once, after train_model.py --final-test.

Usage: python calibrate_rolling.py [--base PATH] [--tag TAG] [--window-days 14] [--test]
Needs calibrate.py in the same folder. Reads results/<tag>val_predictions.parquet.
Writes results/<tag>rolling_check.csv and results/<tag>val_predictions_rolling.parquet
(q10/q90 calibrated, q10_raw/q90_raw kept, calibrated flag).
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

from calibrate import EPS, compare, conformal_q, fit, apply, parts

MIN_N = 300           # resolved forecasts needed in the window before correcting (per horizon)
COMPARE_WINDOWS = (7, 14, 28)


def adjust(d, q_lo, q_hi):
    w = np.maximum(d["q90"] - d["q10"], EPS).to_numpy()
    lo = np.minimum(np.maximum(d["q10"].to_numpy() - q_lo * w, 0.0), d["q50"].to_numpy())
    hi = np.maximum(d["q90"].to_numpy() + q_hi * w, d["q50"].to_numpy())
    return d.assign(q10=lo, q90=hi)


def rolling(d, window_days, min_n=MIN_N):
    """d must have a default RangeIndex. Returns (adjusted frame, boolean mask of corrected rows)."""
    s_lo, s_hi, _ = parts(d)
    q_lo, q_hi = np.zeros(len(d)), np.zeros(len(d))
    done = np.zeros(len(d), dtype=bool)
    day = d["origin_ts"].dt.normalize()
    span = np.timedelta64(window_days, "D")
    for _, g in d.groupby("horizon"):
        gs = g.sort_values("target_ts")
        tt = gs["target_ts"].to_numpy()
        slo, shi = s_lo.loc[gs.index].to_numpy(), s_hi.loc[gs.index].to_numpy()
        for D, rows in g.groupby(day.loc[g.index]).groups.items():
            T = np.datetime64(D)
            hi_i = np.searchsorted(tt, T, side="right")        # target_ts <= start of day D
            lo_i = np.searchsorted(tt, T - span, side="right")  # target_ts  > D - window
            assert hi_i == 0 or tt[hi_i - 1] <= T
            if hi_i - lo_i < min_n:
                continue
            idx = np.asarray(rows)
            q_lo[idx] = conformal_q(slo[lo_i:hi_i])
            q_hi[idx] = conformal_q(shi[lo_i:hi_i])
            done[idx] = True
    return adjust(d, q_lo, q_hi), done


def summarize(fold, variant, raw, cal, share=1.0):
    below, above = float((cal["y"] < cal["q10"]).mean()), float((cal["y"] > cal["q90"]).mean())
    ratio = float((cal["q90"] - cal["q10"]).mean() / (raw["q90"] - raw["q10"]).mean())
    return dict(fold=fold, variant=variant, coverage=1 - below - above, below10=below, above10=above,
                width_ratio=ratio, calibrated_share=share)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--tag", default="")
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--test", action="store_true", help="also score the test predictions (use once)")
    args, _ = ap.parse_known_args()  # ignore Colab's injected "-f kernel.json"
    res = pathlib.Path(args.base) / "results"

    val = pd.read_parquet(res / f"{args.tag}val_predictions.parquet")
    folds = sorted(val["fold"].unique())
    allp = val
    if args.test:
        f = res / "test_predictions.parquet"
        assert f.exists(), "run train_model.py --final-test first"
        print("WARNING: scoring the held-out TEST window. Use once.")
        allp = pd.concat([val, pd.read_parquet(f).assign(fold="test")], ignore_index=True)
        folds = folds + ["test"]
    allp = allp.reset_index(drop=True)
    for a, b in zip(folds, folds[1:]):
        assert allp.loc[allp.fold == a, "target_ts"].max() < allp.loc[allp.fold == b, "target_ts"].min(), \
            "folds overlap in time"

    windows = sorted(set(COMPARE_WINDOWS) | {args.window_days})
    results = {w: rolling(allp, w) for w in windows}
    static_q = {}
    summ = []
    for i, f in enumerate(folds):
        m = (allp.fold == f).to_numpy()
        raw = allp[m]
        summ.append(summarize(f, "raw", raw, raw))
        if i > 0:
            prior = allp[allp.fold.isin(folds[:i])] if f != "test" else allp[allp.fold != "test"]
            summ.append(summarize(f, "static (earlier folds)", raw, apply(raw, fit(prior))))
        for w in windows:
            cal, done = results[w]
            summ.append(summarize(f, f"rolling {w}d", raw, cal[m], float(done[m].mean())))
    table = pd.DataFrame(summ)
    table.to_csv(res / f"{args.tag}rolling_check.csv", index=False)
    for f in folds:
        print(f"\n=== {f}: all horizons pooled (target: coverage 0.80, tails 0.10 / 0.10) ===")
        print(table[table.fold == f].drop(columns="fold").round(3).to_string(index=False))

    w = args.window_days
    cal, done = results[w]
    for f in folds:
        m = (allp.fold == f).to_numpy()
        compare(f, allp[m], cal[m], f"rolling {w}d")
    out = cal.assign(q10_raw=allp["q10"], q90_raw=allp["q90"], calibrated=done)
    out[out.fold != "test"].to_parquet(res / f"{args.tag}val_predictions_rolling.parquet", index=False)
    msg = f"\nsaved {res / (args.tag + 'rolling_check.csv')} and {res / (args.tag + 'val_predictions_rolling.parquet')}"
    if args.test:
        out[out.fold == "test"].to_parquet(res / "test_predictions_rolling.parquet", index=False)
        msg += f" and {res / 'test_predictions_rolling.parquet'}"
    print(msg)


if __name__ == "__main__":
    main()
