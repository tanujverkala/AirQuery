"""LightGBM multi-horizon quantile model for AirQuery (10 / 50 / 90 % quantiles of PM2.5).

One model per quantile, trained on the stacked table from build_features.py with
`horizon` as a feature. Validation uses the rolling-origin folds from baselines.py.

Validation run (default):
  1. Tune on the MEDIAN (q50) only: each config is fit on every fold, and scored as
     mean over horizons of MAE_model / MAE_best_naive (best naive = the better of
     persistence and same-hour-yesterday at that horizon). Mean over the 3 folds picks
     the config.
  2. Fit q10 and q90 with the chosen config on every fold.
  3. Report by horizon: MAE, RMSE, skill vs persistence and vs best naive, 80%
     interval coverage and width, poor-air event precision / recall.
The held-out test window is NOT touched unless you pass --final-test (use once).

Defaults (--drop doy_sin,doy_cos,dow --target delta): with ~2 years of data, day-of-year and
day-of-week let the trees memorise specific dates, and fold 1 validates on months absent from
training, which trees cannot extrapolate to. The delta target (PM2.5 at t+h minus PM2.5 at t,
added back afterwards) makes the strong persistence signal easy to learn. Both were chosen on
synthetic data only; compare on the real data with
  python train_model.py --configs B --drop none --target level --tag raw_
(--tag keeps the variant's result files separate).

Every fit is cached in results/cache/, so an interrupted run resumes and a finished
run re-prints instantly. Changing a config, the feature list or --quick makes new
cache keys, so stale results are never reused.

Usage:
  python train_model.py --quick                    # smoke test, minutes, NOT for reporting
  python train_model.py                            # full validation (configs A,B,C)
  python train_model.py --configs A,B              # fewer configs
  python train_model.py --configs B --target logratio --tag logr_   # log-space variant
  python train_model.py --final-test               # once, after tuning is finished
  (--final-test reads features and target from results/best_config.json, pass the same --tag)
"""
import argparse
import hashlib
import json
import pathlib
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import FINAL, FOLDS, mae, rmse, split

SEED = 42
QUANTILES = (0.1, 0.5, 0.9)
DROP_DEFAULT = "doy_sin,doy_cos,dow"  # calendar position lets trees memorise dates; see docstring
POOR_DEFAULT = 90.0   # ug/m3; CPCB "Poor" band starts at 91 (24-h basis, secondary source); applied to hourly values here
SHOW = [3, 12, 24, 36, 48]
META = ["station", "origin_ts", "target_ts", "target_pm25"]
CAT = "station_id"

COMMON = dict(feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0)
CONFIGS = {
    "A": dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, n_rounds=400, **COMMON),
    "B": dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, n_rounds=400, **COMMON),
    "C": dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200, n_rounds=600, **COMMON),
}


# ---------------------------------------------------------------- data
def load(base):
    p = base / "data/features/features.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(base / "data/features/features.csv.gz", parse_dates=["origin_ts", "target_ts"])


def lgb_params(cfg, alpha):
    return dict(objective="quantile", alpha=alpha, learning_rate=cfg["learning_rate"],
                num_leaves=cfg["num_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"],
                feature_fraction=cfg["feature_fraction"], bagging_fraction=cfg["bagging_fraction"],
                bagging_freq=cfg["bagging_freq"], lambda_l2=cfg["lambda_l2"],
                seed=SEED, deterministic=True, force_row_wise=True, verbosity=-1)


class Split:
    """One train/eval split; the LightGBM Dataset is built lazily, only if a fit is needed."""

    def __init__(self, name, spec, df, X, y, features, quick, target="level"):
        tr, ev = split(df, spec)
        tr_idx = np.flatnonzero(tr.to_numpy())
        lag0 = df["pm25_lag0"].to_numpy(np.float32)
        if target in ("delta", "logratio"):   # both need PM2.5 at the origin hour
            tr_idx = tr_idx[~np.isnan(lag0[tr_idx])]
        if target == "delta":          # label = PM2.5 at target minus PM2.5 at origin
            y = y - lag0
        elif target == "logratio":     # label = log(1+PM2.5 at target) - log(1+PM2.5 at origin)
            y = (np.log1p(y.astype(np.float64)) - np.log1p(lag0.astype(np.float64))).astype(np.float32)
        if quick:
            tr_idx = tr_idx[::5]
        self.name, self.features, self.quick, self.target = name, features, quick, target
        self.tr_idx, self.ev_idx = tr_idx, np.flatnonzero(ev.to_numpy())
        self.X, self.y = X, y
        self.ev = df.iloc[self.ev_idx][["station", "origin_ts", "target_ts", "horizon",
                                       "target_pm25", "pm25_lag0", "base_yesterday"]].reset_index(drop=True)
        self.ev = self.ev.rename(columns={"target_pm25": "y", "pm25_lag0": "persistence",
                                          "base_yesterday": "yesterday"})
        self._ds = None

    def dataset(self):
        if self._ds is None:
            self._ds = lgb.Dataset(self.X[self.tr_idx], label=self.y[self.tr_idx],
                                   feature_name=self.features, categorical_feature=[CAT],
                                   params=dict(feature_pre_filter=False, max_bin=255, verbosity=-1),
                                   free_raw_data=False)
        return self._ds


# ---------------------------------------------------------------- fitting with cache
def predict(booster, sp):
    p = booster.predict(sp.X[sp.ev_idx]).astype(np.float32)
    if sp.target == "delta":           # add the origin-time PM2.5 back (NaN where it is missing)
        p = p + sp.ev["persistence"].to_numpy(np.float32)
    elif sp.target == "logratio":      # quantiles map through the monotone transform unchanged
        lag0 = sp.ev["persistence"].to_numpy(np.float64)
        p = np.expm1(np.log1p(lag0) + p.astype(np.float64)).astype(np.float32)
    return p


def cache_key(cfg_name, cfg, sp, q):
    blob = json.dumps([cfg, sp.name, q, len(sp.tr_idx), sp.quick, sp.features, sp.target, SEED],
                      sort_keys=True)
    return f"{cfg_name}_{sp.name}_q{int(round(q * 100))}_{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


def get_pred(cfg_name, cfg, sp, q, cache):
    key = cache_key(cfg_name, cfg, sp, q)
    f_pred, f_imp = cache / f"{key}.npy", cache / f"{key}_imp.csv"
    if f_pred.exists() and f_imp.exists():
        return np.load(f_pred)
    rounds = min(cfg["n_rounds"], 60) if sp.quick else cfg["n_rounds"]
    t0 = time.time()
    booster = lgb.train(lgb_params(cfg, q), sp.dataset(), num_boost_round=rounds)
    pred = predict(booster, sp)
    gain = booster.feature_importance("gain")
    pd.DataFrame({"feature": sp.features, "gain_pct": 100 * gain / gain.sum()}).to_csv(f_imp, index=False)
    np.save(f_pred, pred)
    print(f"   fit {cfg_name} {sp.name} q{int(q * 100)}: {rounds} trees, "
          f"{len(sp.tr_idx)} rows, {time.time() - t0:.0f}s", flush=True)
    return pred


# ---------------------------------------------------------------- evaluation
def event_scores(y, alarms, thr):
    event = y > thr
    rows = []
    for name, a in alarms.items():
        tp, fp, fn = int((a & event).sum()), int((a & ~event).sum()), int((~a & event).sum())
        rows.append(dict(rule=name, event_rate=float(event.mean()), alarms=int(a.sum()),
                         precision=tp / (tp + fp) if tp + fp else np.nan,
                         recall=tp / (tp + fn) if tp + fn else np.nan))
    return pd.DataFrame(rows)


def evaluate(name, ev, P, thr):
    """ev: eval frame; P: (3, n) quantile predictions already sorted. Common rows only."""
    d = ev.copy()
    d["q10"], d["q50"], d["q90"] = P[0], P[1], P[2]
    d = d[d[["persistence", "yesterday"]].notna().all(axis=1)]
    recs = []
    for h, g in d.groupby("horizon"):
        m_p, m_y = mae(g["y"], g["persistence"]), mae(g["y"], g["yesterday"])
        m_best, m = min(m_p, m_y), mae(g["y"], g["q50"])
        below, above = float((g["y"] < g["q10"]).mean()), float((g["y"] > g["q90"]).mean())
        recs.append(dict(fold=name, horizon=int(h), n=len(g), mae_model=m, rmse_model=rmse(g["y"], g["q50"]),
                         mae_persistence=m_p, mae_yesterday=m_y, mae_best_naive=m_best,
                         skill_vs_persistence=1 - m / m_p, skill_vs_best_naive=1 - m / m_best,
                         coverage80=1 - below - above, below10=below, above10=above,
                         mean_width=float((g["q90"] - g["q10"]).mean())))
    by_h = pd.DataFrame(recs)
    alarms = {"model_q50": (d["q50"] > thr).to_numpy(), "model_q90": (d["q90"] > thr).to_numpy(),
              "persistence": (d["persistence"] > thr).to_numpy(), "yesterday": (d["yesterday"] > thr).to_numpy()}
    events = event_scores(d["y"].to_numpy(), alarms, thr)
    events.insert(0, "fold", name)
    return by_h, events, d


def show(name, by_h, events, thr, mean_pm25):
    print(f"\n=== {name} | eval rows (common) {int(by_h['n'].sum())} | mean PM2.5 {mean_pm25:.1f} ===")
    h = [x for x in SHOW if x in set(by_h["horizon"])]
    t = by_h.set_index("horizon").loc[h]
    out = pd.DataFrame({"model_MAE": t["mae_model"], "best_naive_MAE": t["mae_best_naive"],
                        "skill_vs_best_naive": t["skill_vs_best_naive"],
                        "skill_vs_persist": t["skill_vs_persistence"],
                        "coverage80": t["coverage80"], "below10": t["below10"], "above10": t["above10"],
                        "width": t["mean_width"]}).round(2)
    print(out.T.to_string())
    print(f"Events: PM2.5 > {thr:g} ug/m3 (pooled over horizons)")
    print(events.drop(columns="fold").round(3).to_string(index=False))


def sorted_quantiles(p10, p50, p90):
    P = np.vstack([p10, p50, p90])
    crossing = float(((P[0] > P[1]) | (P[1] > P[2])).mean())
    return np.maximum(np.sort(P, axis=0), 0.0), crossing


def selection_score(sp, p50):
    ev = sp.ev.copy()
    ev["p"] = p50
    ev = ev[ev[["persistence", "yesterday"]].notna().all(axis=1)]
    r = []
    for _, g in ev.groupby("horizon"):
        best = min(mae(g["y"], g["persistence"]), mae(g["y"], g["yesterday"]))
        r.append(mae(g["y"], g["p"]) / best)
    return float(np.mean(r))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--configs", default="A,B,C")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1/5 of rows, <=60 trees")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--poor", type=float, default=POOR_DEFAULT)
    ap.add_argument("--drop", default=DROP_DEFAULT,
                    help="comma-separated feature names to leave out; 'none' keeps every feature")
    ap.add_argument("--target", default="delta", choices=["level", "delta", "logratio"],
                    help="delta: PM2.5 minus PM2.5 at the origin hour; level: PM2.5 itself; "
                         "logratio: log(1+PM2.5) minus log(1+PM2.5 at the origin hour), so errors scale with the level")
    ap.add_argument("--tag", default="", help="prefix for result files, to keep variants side by side")
    args, _ = ap.parse_known_args()  # ignore Colab's injected "-f kernel.json"
    base = pathlib.Path(args.base)
    res, cache = base / "results", base / "results/cache"
    cache.mkdir(parents=True, exist_ok=True)
    tag = args.tag + ("quick_" if args.quick else "")

    df = load(base)
    if args.final_test:   # reuse exactly the settings that were validated
        f = res / f"{args.tag}best_config.json"
        assert f.exists(), f"run the full validation first ({f} is missing)"
        bc = json.load(open(f))
        assert not bc.get("quick"), "best_config.json comes from a --quick run; run the full validation first"
        features, args.target = bc["features"], bc["target"]
    else:
        drop = [c.strip() for c in args.drop.split(",") if c.strip() and c.strip().lower() != "none"]
        unknown = [c for c in drop if c not in df.columns]
        assert not unknown, f"--drop names not in the table: {unknown}"
        features = [c for c in df.columns if c not in META and c not in drop]
    assert CAT in features and "horizon" in features
    X = df[features].to_numpy(np.float32)
    y = df["target_pm25"].to_numpy(np.float32)
    print(f"rows {len(df)} | features {len(features)} | target {args.target} | lightgbm {lgb.__version__}"
          + (" | QUICK MODE (not for reporting)" if args.quick else ""))

    if args.final_test:
        run_final(args, df, X, y, features, res, bc)
        return

    names = [c.strip() for c in args.configs.split(",")]
    assert all(c in CONFIGS for c in names), f"configs must be among {list(CONFIGS)}"
    splits = {n: Split(n, spec, df, X, y, features, args.quick, args.target) for n, spec in FOLDS.items()}

    # 1) tune on the median
    sel, p50 = [], {}
    for c in names:
        for n, sp in splits.items():
            p50[(c, n)] = get_pred(c, CONFIGS[c], sp, 0.5, cache)
            sel.append(dict(config=c, fold=n, score=selection_score(sp, p50[(c, n)])))
    sel = pd.DataFrame(sel)
    tab = sel.pivot(index="config", columns="fold", values="score")
    tab["mean"] = tab.mean(axis=1)
    print("\nConfig selection: mean over horizons of MAE_model / MAE_best_naive (q50; lower is better)")
    print(tab.round(3).to_string())
    best = tab["mean"].idxmin()
    print(f"-> best config: {best}")
    res.mkdir(exist_ok=True)
    sel.to_csv(res / f"{tag}config_selection.csv", index=False)
    json.dump(dict(config=best, params=CONFIGS[best], quick=args.quick, seed=SEED, target=args.target,
                   features=features, score=float(tab.loc[best, "mean"])),
              open(res / f"{tag}best_config.json", "w"), indent=2)

    # 2) intervals with the best config, then 3) evaluation
    all_h, all_e, preds = [], [], []
    for n, sp in splits.items():
        p10 = get_pred(best, CONFIGS[best], sp, 0.1, cache)
        p90 = get_pred(best, CONFIGS[best], sp, 0.9, cache)
        P, crossing = sorted_quantiles(p10, p50[(best, n)], p90)
        by_h, events, d = evaluate(n, sp.ev, P, args.poor)
        show(n, by_h, events, args.poor, float(sp.ev["y"].mean()))
        print(f"quantile crossing before sorting: {100 * crossing:.2f}% of rows")
        all_h.append(by_h)
        all_e.append(events)
        preds.append(d.assign(fold=n))
    pd.concat(all_h).to_csv(res / f"{tag}model_val_metrics.csv", index=False)
    pd.concat(all_e).to_csv(res / f"{tag}model_val_events.csv", index=False)
    pd.concat(preds).to_parquet(res / f"{tag}val_predictions.parquet", index=False)

    last = list(splits.values())[-1]
    imp = pd.read_csv(cache / f"{cache_key(best, CONFIGS[best], last, 0.5)}_imp.csv")
    imp.sort_values("gain_pct", ascending=False).to_csv(res / f"{tag}feature_importance.csv", index=False)
    print("\nTop 10 features by gain (% , median model, last fold):")
    print(imp.sort_values("gain_pct", ascending=False).head(10).round(1).to_string(index=False))
    write_info(res, tag, features, args)
    print(f"\nsaved to {res}/ ({tag}model_val_metrics.csv, {tag}model_val_events.csv, "
          f"{tag}val_predictions.parquet, {tag}feature_importance.csv, {tag}best_config.json)")


def run_final(args, df, X, y, features, res, bc):
    print("WARNING: scoring the held-out TEST window (Oct-Dec 2025). Use this once, after all tuning.")
    cfg = bc["params"]
    print("using config", bc["config"], cfg, "| target", args.target, "|", len(features), "features")
    sp = Split("test", FINAL, df, X, y, features, False, args.target)
    models = res.parent / "models"
    models.mkdir(exist_ok=True)
    preds = []
    for q in QUANTILES:
        t0 = time.time()
        booster = lgb.train(lgb_params(cfg, q), sp.dataset(), num_boost_round=cfg["n_rounds"])
        booster.save_model(str(models / f"lgbm_q{int(round(q * 100))}.txt"))
        preds.append(predict(booster, sp))
        print(f"   fit q{int(q * 100)}: {len(sp.tr_idx)} rows, {time.time() - t0:.0f}s", flush=True)
        if q == 0.5:
            gain = booster.feature_importance("gain")
            pd.DataFrame({"feature": features, "gain_pct": 100 * gain / gain.sum()}).sort_values(
                "gain_pct", ascending=False).to_csv(res / "final_feature_importance.csv", index=False)
    P, crossing = sorted_quantiles(*preds)
    by_h, events, d = evaluate("test", sp.ev, P, args.poor)
    show("test", by_h, events, args.poor, float(sp.ev["y"].mean()))
    print(f"quantile crossing before sorting: {100 * crossing:.2f}% of rows")
    by_h.to_csv(res / "model_test_metrics.csv", index=False)
    events.to_csv(res / "model_test_events.csv", index=False)
    d.to_parquet(res / "test_predictions.parquet", index=False)
    json.dump(dict(features=features, target=args.target, config=bc["config"], params=cfg,
                   quantiles=list(QUANTILES), seed=SEED, train_window=FINAL[:2], test_window=FINAL[1:],
                   postprocess="prediction = model output (+ pm25_lag0 if target is delta; "
                               "expm1(log1p(pm25_lag0) + output) if target is logratio); "
                               "then sort q10/q50/q90 per row and clip at 0"),
              open(models / "model_info.json", "w"), indent=2)
    write_info(res, "final_", features, args)
    print("\nsaved models to", models, "and test results to", res)


def write_info(res, tag, features, args):
    json.dump(dict(lightgbm=lgb.__version__, python=platform.python_version(), numpy=np.__version__,
                   pandas=pd.__version__, seed=SEED, quick=args.quick, poor_threshold=args.poor,
                   target=args.target, configs=CONFIGS, features=features), open(res / f"{tag}run_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
