"""Collect the result files the AirQuery website needs into airquery_app/data/.

Run it once in Colab from the project folder, after train_model.py --final-test,
calibrate_rolling.py --test and error_analysis.py (both with and without --test):
    !python prepare_app_data.py

Output (all under airquery_app/data/):
  test_predictions.parquet   trimmed, float32: the calibrated test-window forecasts
  *.csv                      metrics, calibration, importance, error-analysis tables
  figures/val, figures/test  PNG figures
  ablation.csv               the four model variants you ran (numbers copied from your printed outputs)
  meta.json                  window dates, row counts, versions, creation time
Missing optional files are listed and skipped; the website hides what it cannot find.
"""
import argparse
import json
import pathlib
import platform
import shutil
import sys

import numpy as np
import pandas as pd

KEEP = ["station", "origin_ts", "target_ts", "horizon", "y", "persistence", "yesterday", "q10", "q50", "q90"]

# Mean ratio of model MAE to best-naive MAE over horizons (lower is better), config B, validation folds.
# Copied from the outputs printed by train_model.py; the first row is the frozen setup.
ABLATION = [
    ("Frozen setup: no calendar features, delta target, forecast weather", 0.954, 0.774, 0.838, 0.855),
    ("Log-ratio target instead of delta", 0.954, 0.769, 0.829, 0.851),
    ("Without forecast-weather features", 1.003, 0.865, 0.871, 0.913),
    ("Calendar features kept, level target", 1.637, 0.816, 0.856, 1.103),
]

CSV_TOP = ["model_test_metrics.csv", "model_test_events.csv", "model_val_metrics.csv", "model_val_events.csv",
           "rolling_check.csv", "final_feature_importance.csv", "feature_importance.csv", "baselines_val.csv"]
CSV_ERR = ["by_station.csv", "by_hour_block.csv", "by_forecast_level.csv", "worst_days.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--out", default="airquery_app/data")
    args, _ = ap.parse_known_args()  # ignore Colab's injected "-f kernel.json"
    base, out = pathlib.Path(args.base), pathlib.Path(args.out)
    res = base / "results"
    out.mkdir(parents=True, exist_ok=True)
    missing = []

    src = res / "test_predictions_rolling.parquet"
    if not src.exists():
        sys.exit(f"{src} not found. Run train_model.py --final-test and calibrate_rolling.py --test first.")
    d = pd.read_parquet(src)[KEEP].sort_values(["station", "origin_ts", "horizon"]).reset_index(drop=True)
    num = d.select_dtypes("float64").columns
    d[num] = d[num].astype("float32")
    d.to_parquet(out / "test_predictions.parquet", index=False)
    print(f"test_predictions.parquet: {len(d)} rows, {(out / 'test_predictions.parquet').stat().st_size / 1e6:.1f} MB")

    for f in CSV_TOP:
        if (res / f).exists():
            shutil.copy(res / f, out / f)
        else:
            missing.append(f)
    for folder, tag in [("error_analysis", "val_error"), ("test_error_analysis", "test_error")]:
        (out / tag).mkdir(exist_ok=True)
        for f in CSV_ERR:
            if (res / folder / f).exists():
                shutil.copy(res / folder / f, out / tag / f)
            else:
                missing.append(f"{folder}/{f}")
    for folder, tag in [("figures", "val"), ("test_figures", "test")]:
        (out / "figures" / tag).mkdir(parents=True, exist_ok=True)
        pngs = sorted((res / folder).glob("*.png"))
        if not pngs:
            missing.append(f"{folder}/*.png")
        for p in pngs:
            shutil.copy(p, out / "figures" / tag / p.name)

    pd.DataFrame(ABLATION, columns=["variant", "fold1", "fold2", "fold3", "mean"]).to_csv(out / "ablation.csv", index=False)
    meta = dict(created=pd.Timestamp.now().isoformat(timespec="seconds"), rows=int(len(d)),
                stations=sorted(d["station"].unique()), horizons=sorted(int(h) for h in d["horizon"].unique()),
                first_target=str(d["target_ts"].min()), last_target=str(d["target_ts"].max()),
                first_origin=str(d["origin_ts"].min()), last_origin=str(d["origin_ts"].max()),
                python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__)
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print("meta.json written; total size of data folder: "
          f"{sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6:.1f} MB")
    if missing:
        print("not found (skipped):", ", ".join(missing))


if __name__ == "__main__":
    main()
