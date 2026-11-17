"""Interval calibration for AirQuery: split-conformal, one correction per tail, per horizon.

The raw intervals [q10, q90] cover less than 80% of outcomes. For every horizon this finds
how far outside (or inside) the interval the outcomes land, in units of the predicted
width w = max(q90 - q10, EPS), so one correction works in winter and in monsoon alike:
    lower tail:  s_lo = (q10 - y) / w     upper tail:  s_hi = (y - q90) / w
    Q_lo, Q_hi = the conformal (1 - TAIL) quantile of s_lo, s_hi    (TAIL = 0.10)
    calibrated interval = [q10 - Q_lo * w, q90 + Q_hi * w], with lo <= q50 <= hi and lo >= 0.
A negative Q shrinks that side.

Honest out-of-time check: fold k is calibrated ONLY with the validation scores of the folds
before it (fold1 has nothing earlier, so it is reported uncalibrated). The table saved for
the website uses all validation folds. The test window is touched only with --test, after
train_model.py --final-test has been run once.

Usage: python calibrate.py [--base PATH] [--tag TAG] [--test]
Reads results/<tag>val_predictions.parquet, writes results/<tag>calibration.json
and results/<tag>calibration_check.csv.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

EPS = 1.0      # ug/m3, floor on the predicted width so w never reaches 0 (engineering choice)
TAIL = 0.10    # 10% below and 10% above, so 80% inside
SHOW = [3, 12, 24, 36, 48]


def parts(d):
    w = np.maximum(d["q90"] - d["q10"], EPS)
    return (d["q10"] - d["y"]) / w, (d["y"] - d["q90"]) / w, w


def conformal_q(s):
    n = len(s)
    level = min(1.0, np.ceil((n + 1) * (1 - TAIL)) / n)
    return float(np.quantile(s, level, method="higher"))


def fit(d):
    s_lo, s_hi, _ = parts(d)
    rows = []
    for h, idx in d.groupby("horizon").groups.items():
        rows.append(dict(horizon=int(h), n=len(idx), q_lo=conformal_q(s_lo.loc[idx]),
                         q_hi=conformal_q(s_hi.loc[idx])))
    return pd.DataFrame(rows).set_index("horizon")


def apply(d, Q):
    w = np.maximum(d["q90"] - d["q10"], EPS)
    q = Q.reindex(d["horizon"].to_numpy())
    assert not q.isna().any().any(), "a horizon in the data has no calibration entry"
    lo = np.minimum(np.maximum(d["q10"].to_numpy() - q["q_lo"].to_numpy() * w, 0.0), d["q50"].to_numpy())
    hi = np.maximum(d["q90"].to_numpy() + q["q_hi"].to_numpy() * w, d["q50"].to_numpy())
    return d.assign(q10=lo, q90=hi)


def coverage(d):
    below, above = d["y"] < d["q10"], d["y"] > d["q90"]
    return pd.DataFrame({"coverage80": 1 - below.groupby(d["horizon"]).mean() - above.groupby(d["horizon"]).mean(),
                         "below10": below.groupby(d["horizon"]).mean(),
                         "above10": above.groupby(d["horizon"]).mean(),
                         "width": (d["q90"] - d["q10"]).groupby(d["horizon"]).mean()})


def compare(name, raw, cal, source):
    a, b = coverage(raw), coverage(cal)
    out = pd.concat({"raw": a, "calibrated": b}, axis=1)
    out.insert(0, "set", name)
    out.insert(1, "calibrated_with", source)
    h = [x for x in SHOW if x in a.index]
    t = pd.DataFrame({"coverage raw": a.loc[h, "coverage80"], "coverage cal": b.loc[h, "coverage80"],
                      "below10 cal": b.loc[h, "below10"], "above10 cal": b.loc[h, "above10"],
                      "width raw": a.loc[h, "width"], "width cal": b.loc[h, "width"]}).round(2)
    print(f"\n=== {name} (calibrated with: {source}) ===")
    print(t.T.to_string())
    print(f"mean coverage over all horizons: raw {a['coverage80'].mean():.3f} -> calibrated {b['coverage80'].mean():.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--tag", default="")
    ap.add_argument("--test", action="store_true", help="also calibrate the test predictions (use once)")
    args, _ = ap.parse_known_args()  # ignore Colab's injected "-f kernel.json"
    res = pathlib.Path(args.base) / "results"

    val = pd.read_parquet(res / f"{args.tag}val_predictions.parquet").reset_index(drop=True)
    folds = sorted(val["fold"].unique())
    spans = [(val.loc[val.fold == f, "target_ts"].min(), val.loc[val.fold == f, "target_ts"].max()) for f in folds]
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end < start, "folds overlap in time; calibration would not be out-of-time"

    checks = [compare(folds[0], val[val.fold == folds[0]], val[val.fold == folds[0]], "nothing (no earlier fold)")]
    for i in range(1, len(folds)):
        prior = val[val.fold.isin(folds[:i])]
        cur = val[val.fold == folds[i]]
        checks.append(compare(folds[i], cur, apply(cur, fit(prior)), "+".join(folds[:i])))
    pd.concat(checks).to_csv(res / f"{args.tag}calibration_check.csv")

    Q = fit(val)
    chk = coverage(apply(val, Q))
    print(f"\nin-sample sanity check (calibration data itself): coverage {chk['coverage80'].mean():.3f}, "
          f"below {chk['below10'].mean():.3f}, above {chk['above10'].mean():.3f} (should be about 0.80 / 0.10 / 0.10)")
    json.dump(dict(tail=TAIL, eps=EPS, calibrated_with=folds, rule="lo=q10-q_lo*w, hi=q90+q_hi*w, w=max(q90-q10,eps); "
                   "then lo=min(max(lo,0),q50), hi=max(hi,q50)",
                   horizons={int(h): dict(q_lo=r.q_lo, q_hi=r.q_hi, n=int(r.n)) for h, r in Q.iterrows()}),
              open(res / f"{args.tag}calibration.json", "w"), indent=2)
    print("saved", res / f"{args.tag}calibration.json", "and", res / f"{args.tag}calibration_check.csv")

    if args.test:
        f = res / "test_predictions.parquet"
        assert f.exists(), "run train_model.py --final-test first"
        test = pd.read_parquet(f).reset_index(drop=True)
        print("\nWARNING: scoring the held-out TEST window. Use once.")
        compare("test", test, apply(test, Q), "+".join(folds))
        apply(test, Q).to_parquet(res / "test_predictions_calibrated.parquet", index=False)


if __name__ == "__main__":
    main()
