"""Error analysis for AirQuery on the VALIDATION folds (the test window is not touched).

Reads results/<tag>val_predictions_rolling.parquet (from calibrate_rolling.py) and writes
tables to results/error_analysis/ and four figures to results/figures/:
  by_station.csv   per station and fold: MAE, skill vs naive, bias, coverage raw / calibrated
  by_hour.csv      per hour of the TARGET time and fold
  by_level.csv     per OBSERVED PM2.5 band (CPCB band edges) and fold. This slices on the outcome,
                   so any noisy forecast looks under-forecast at the top and over-forecast at the
                   bottom: do not read bias claims from it alone
  by_forecast_level.csv  per FORECAST (q50) band and fold: bias and coverage conditioned on what the
                   model predicted, which is the fair reliability check
  worst_days.csv   the 10 target dates with the largest mean absolute error in each fold
  fig_mae_by_horizon.png, fig_coverage_by_horizon.png, fig_error_by_level.png,
  fig_example_forecasts.png  (one 24 h forecast week per fold, the week with the MEDIAN
                              weekly MAE for the chosen station: not picked for looks)

"Naive" = for each horizon, whichever of persistence / same-hour-yesterday has the lower
MAE over all validation rows (a fixed rule, so slices compare against the same predictor).
Skill = 1 - MAE_model / MAE_naive. Bias = mean(q50 - y); positive means over-forecast.
Calibrated coverage uses the rolling calibration (warm-up rows are uncorrected).

With --test it describes the TEST predictions instead (results/test_predictions_rolling.parquet,
written by calibrate_rolling.py --test), into results/test_error_analysis and results/test_figures.
Use it for reporting only: the model, features and calibration are frozen once the test is scored.

Usage: python error_analysis.py [--base PATH] [--tag TAG] [--station "DTU, Delhi - CPCB"] [--test]
"""
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BANDS = [0, 30, 60, 90, 120, 250, np.inf]   # CPCB PM2.5 band edges (secondary source), ug/m3
BAND_LABELS = ["0-30", "31-60", "61-90", "91-120", "121-250", ">250"]
BLOCKS = {"00-05": range(0, 6), "06-11": range(6, 12), "12-17": range(12, 18), "18-23": range(18, 24)}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def panels(n, title, ylabel):
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        style(ax)
    axes[0].set_ylabel(ylabel, color=INK2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    return fig, axes


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--tag", default="")
    ap.add_argument("--station", default="DTU, Delhi - CPCB")
    ap.add_argument("--test", action="store_true", help="describe the test predictions (reporting only)")
    args, _ = ap.parse_known_args()  # ignore Colab's injected "-f kernel.json"
    base = pathlib.Path(args.base)
    res = base / "results"
    if args.test:
        args.tag = "test_"
    tabs, figs = res / f"{args.tag}error_analysis", res / f"{args.tag}figures"  # tag keeps variants apart
    tabs.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    d = pd.read_parquet(res / ("test_predictions_rolling.parquet" if args.test
                               else f"{args.tag}val_predictions_rolling.parquet"))
    d = d.dropna(subset=["y", "q50", "persistence", "yesterday"]).reset_index(drop=True)
    folds = sorted(d["fold"].unique())

    # fixed naive rule per horizon
    by_h = d.groupby("horizon").apply(
        lambda g: pd.Series({"p": mae(g["y"], g["persistence"]), "y": mae(g["y"], g["yesterday"])}),
        include_groups=False)
    use_pers = (by_h["p"] <= by_h["y"]).to_dict()
    d["naive"] = np.where(d["horizon"].map(use_pers), d["persistence"], d["yesterday"])
    d["err"], d["abs_model"], d["abs_naive"] = d["q50"] - d["y"], (d["q50"] - d["y"]).abs(), (d["naive"] - d["y"]).abs()
    d["in_raw"] = (d["y"] >= d["q10_raw"]) & (d["y"] <= d["q90_raw"])
    d["in_cal"] = (d["y"] >= d["q10"]) & (d["y"] <= d["q90"])
    d["hour"] = d["target_ts"].dt.hour
    d["band"] = pd.cut(d["y"], BANDS, labels=BAND_LABELS, right=True, include_lowest=True)
    d["fband"] = pd.cut(d["q50"], BANDS, labels=BAND_LABELS, right=True, include_lowest=True)

    def agg(g):
        return pd.Series({"n": len(g), "mean_pm25": g["y"].mean(), "mae_model": g["abs_model"].mean(),
                          "mae_naive": g["abs_naive"].mean(),
                          "skill": 1 - g["abs_model"].mean() / g["abs_naive"].mean(),
                          "bias": g["err"].mean(), "coverage_raw": g["in_raw"].mean(),
                          "coverage_cal": g["in_cal"].mean()})

    def table(keys, name):
        t = d.groupby(["fold"] + keys, observed=True).apply(agg, include_groups=False).reset_index()
        t.to_csv(tabs / f"{name}.csv", index=False)
        return t

    st = table(["station"], "by_station")
    d["block"] = d["hour"].map({h: b for b, r in BLOCKS.items() for h in r})
    table(["hour"], "by_hour")
    hb = table(["block"], "by_hour_block")
    lv = table(["band"], "by_level")
    fl = table(["fband"], "by_forecast_level")

    print("Naive rule per horizon (True = persistence):", {int(k): bool(v) for k, v in use_pers.items()})
    for title, t, key in [("BY STATION", st, "station"), ("BY TARGET HOUR BLOCK", hb, "block"),
                          ("BY OBSERVED PM2.5 BAND (slices on the outcome: exaggerates high-end under-forecast)", lv, "band"),
                          ("BY FORECAST PM2.5 BAND (q50)", fl, "fband")]:
        print(f"\n=== {title}: skill vs naive | bias (q50 - y) | calibrated coverage ===")
        extra = [("mean_pm25", "obs_mean")] if key == "fband" else []
        for v, name in [("skill", "skill"), ("bias", "bias"), ("coverage_cal", "cov_cal")] + extra:
            p = t.pivot(index=key, columns="fold", values=v).round(2)
            p.columns = [f"{name} {c}" for c in p.columns]
            print(p.to_string() if v != "skill" else p.to_string())
        if key in ("band", "fband"):
            print(t.pivot(index=key, columns="fold", values="n").fillna(0).astype(int).rename(columns=lambda c: f"n {c}").to_string())

    # worst days
    dd = d.groupby(["fold", d["target_ts"].dt.date.rename("target_date")]).apply(
        lambda g: pd.Series({"n": len(g), "mean_pm25": g["y"].mean(), "mae_model": g["abs_model"].mean(),
                             "mae_naive": g["abs_naive"].mean(), "bias": g["err"].mean()}),
        include_groups=False).reset_index()
    dd = dd[dd["n"] >= 500]
    worst = dd.sort_values(["fold", "mae_model"], ascending=[True, False]).groupby("fold").head(10)
    worst.to_csv(tabs / "worst_days.csv", index=False)
    print("\n=== WORST TARGET DAYS (top 5 per fold by mean abs error; all in worst_days.csv) ===")
    print(worst.groupby("fold").head(5).round(1).to_string(index=False))

    # ---------------- figures
    hz = sorted(d["horizon"].unique())
    fig, axes = panels(len(folds), "Median forecast error by horizon: model vs best naive baseline", "MAE (µg/m³)")
    for ax, f in zip(axes, folds):
        g = d[d.fold == f].groupby("horizon")[["abs_model", "abs_naive"]].mean()
        ax.plot(hz, g["abs_naive"], color=ORANGE, lw=2)
        ax.plot(hz, g["abs_model"], color=BLUE, lw=2)
        ax.annotate("naive", (hz[-1], g["abs_naive"].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
        ax.annotate("model", (hz[-1], g["abs_model"].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
        ax.set_title(f"{f} (mean PM2.5 {d[d.fold == f]['y'].mean():.0f})", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("horizon (h)", color=INK2, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.set_xlim(hz[0], hz[-1] + 9)
    fig.legend(handles=[plt.Line2D([], [], color=BLUE, lw=2, label="model (q50)"),
                        plt.Line2D([], [], color=ORANGE, lw=2, label="best naive")],
               loc="upper right", frameon=False, labelcolor=INK2, fontsize=9, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "fig_mae_by_horizon.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = panels(len(folds), "80% interval coverage by horizon: raw vs rolling-calibrated", "share of outcomes inside")
    for ax, f in zip(axes, folds):
        g = d[d.fold == f].groupby("horizon")[["in_raw", "in_cal"]].mean()
        ax.axhline(0.8, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.plot(hz, g["in_raw"], color=ORANGE, lw=2)
        ax.plot(hz, g["in_cal"], color=BLUE, lw=2)
        ax.annotate("raw", (hz[-1], g["in_raw"].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
        ax.annotate("calibrated", (hz[-1], g["in_cal"].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
        ax.set_title(f, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("horizon (h)", color=INK2, fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.set_xlim(hz[0], hz[-1] + 14)
    fig.legend(handles=[plt.Line2D([], [], color=ORANGE, lw=2, label="raw"),
                        plt.Line2D([], [], color=BLUE, lw=2, label="calibrated"),
                        plt.Line2D([], [], color=INK2, lw=1, ls=(0, (4, 3)), label="target 0.80")],
               loc="upper right", frameon=False, labelcolor=INK2, fontsize=9, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "fig_coverage_by_horizon.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = panels(len(folds), "Median forecast bias by forecast PM2.5 band (q50 minus observed)", "mean error (µg/m³)")
    for ax, f in zip(axes, folds):
        t = fl[fl.fold == f].set_index("fband").reindex(BAND_LABELS)
        ax.bar(range(len(BAND_LABELS)), t["bias"].fillna(0), color=BLUE, width=0.6)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(BAND_LABELS)), BAND_LABELS)
        ax.set_title(f, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("forecast (q50) PM2.5 band (µg/m³)", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "fig_error_by_level.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = panels(len(folds), f"24 h forecasts, {args.station}: the median-MAE week of each fold", "PM2.5 (µg/m³)")
    for ax, f in zip(axes, folds):
        g = d[(d.fold == f) & (d.station == args.station) & (d.horizon == 24)].sort_values("target_ts")
        if g.empty:
            ax.set_title(f"{f}: no rows for this station", color=INK2, fontsize=9, loc="left")
            continue
        wk = g.groupby(pd.Grouper(key="target_ts", freq="7D")).filter(lambda x: len(x) >= 140)
        wk = wk.assign(week=wk["target_ts"].dt.floor("D") - pd.to_timedelta(
            (wk["target_ts"].dt.floor("D") - wk["target_ts"].min().floor("D")).dt.days % 7, unit="D"))
        m = wk.groupby("week")["abs_model"].mean().sort_values()
        pick = m.index[len(m) // 2]
        g = wk[wk.week == pick]
        ax.fill_between(g["target_ts"], g["q10"], g["q90"], color=BLUE, alpha=0.22, lw=0)
        ax.plot(g["target_ts"], g["y"], color=INK, lw=1.6)
        ax.plot(g["target_ts"], g["q50"], color=BLUE, lw=2)
        ax.set_title(f"{f}: week from {pick:%d %b %Y} (MAE {m.loc[pick]:.0f})", color=INK, fontsize=10, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.legend(handles=[plt.Line2D([], [], color=INK, lw=1.6, label="observed"),
                        plt.Line2D([], [], color=BLUE, lw=2, label="forecast (q50)"),
                        plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.22, lw=0, label="80% interval (calibrated)")],
               loc="upper right", frameon=False, labelcolor=INK2, fontsize=9, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "fig_example_forecasts.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nsaved tables to {tabs}/ and figures to {figs}/")


if __name__ == "__main__":
    main()
