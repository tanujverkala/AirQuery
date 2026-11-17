"""AirQuery: 48-hour PM2.5 forecasts with uncertainty for Delhi air quality stations.

Reads only precomputed files from ./data (made by prepare_app_data.py), so the free tier
never has to train or download anything. Run locally with:  streamlit run app.py
"""
import json
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8e8d87"
BLUE_FILL = "rgba(42,120,214,0.20)"
# CPCB PM2.5 categories (24-hour breakpoints, ug/m3; taken from a secondary source, applied here to hourly values)
CPCB = [(30, "Good"), (60, "Satisfactory"), (90, "Moderately polluted"), (120, "Poor"),
        (250, "Very poor"), (np.inf, "Severe")]
STATIONS = {  # coordinates from a published table, 3 decimals; not yet checked against CPCB
    "DTU, Delhi - CPCB": (28.749, 77.120),
    "IHBAS, Dilshad Garden, Delhi - CPCB": (28.681, 77.303),
    "ITO, Delhi - CPCB": (28.629, 77.241),
    "NSIT Dwarka, Delhi - CPCB": (28.609, 77.033),
    "Shadipur, Delhi - CPCB": (28.651, 77.147),
    "Sirifort, Delhi - CPCB": (28.550, 77.216),
    "Wazirpur, Delhi - DPCC": (28.700, 77.165),
}
DQ = {  # PM2.5 coverage % and longest PM2.5 gap (hours), Jan 2024 to Dec 2025, from the cleaning report
    "DTU, Delhi - CPCB": (94.6, 74), "IHBAS, Dilshad Garden, Delhi - CPCB": (93.9, 118),
    "ITO, Delhi - CPCB": (94.7, 147), "NSIT Dwarka, Delhi - CPCB": (95.8, 23),
    "Shadipur, Delhi - CPCB": (94.3, 118), "Sirifort, Delhi - CPCB": (83.7, 2320),
    "Wazirpur, Delhi - DPCC": (97.9, 35),
}
PAGES = ["Problem and Data", "Forecast Explorer", "Best Time to Go Out", "Results", "Conclusions and Limitations"]

st.set_page_config(page_title="AirQuery", page_icon="🌫️", layout="wide")


# ------------------------------------------------------------------ helpers
def category(v):
    for hi, name in CPCB:
        if v <= hi:
            return name
    return CPCB[-1][1]


def short(name):
    return name.split(",")[0]


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def image(path, caption=None):
    try:
        st.image(str(path), caption=caption, width="stretch")
    except Exception:
        st.image(str(path), caption=caption, use_column_width=True)


@st.cache_data
def read_csv(rel):
    p = DATA / rel
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def read_preds():
    p = DATA / "test_predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def read_meta():
    p = DATA / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def weighted_bias():
    """Mean (median forecast - observed) over test rows whose forecast was above 60 ug/m3, or None."""
    err = read_csv("test_error/by_forecast_level.csv")
    if err is None:
        return None
    hb = err[err["fband"].isin(["61-90", "91-120", "121-250", ">250"])]
    return float((hb["bias"] * hb["n"]).sum() / hb["n"].sum()) if len(hb) and hb["n"].sum() > 0 else None


def rolling_coverage():
    """{fold: coverage of the rolling 14-day calibrated 80% range}, or {}."""
    rc = read_csv("rolling_check.csv")
    if rc is None:
        return {}
    r = rc[rc["variant"] == "rolling 14d"]
    return dict(zip(r["fold"], r["coverage"]))


def when(ts):
    return pd.Timestamp(ts).strftime("%a %d %b, %H:%M")


def style_fig(fig, ytitle, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title=ytitle, legend=dict(orientation="h", y=1.1, x=0))
    return fig


# ------------------------------------------------------------------ pages
def page_problem(d):
    st.title("AirQuery")
    st.subheader("48-hour PM2.5 forecasts with uncertainty for Delhi air quality stations")
    st.write(
        "Delhi's air changes hour by hour, and so does what it costs to be outside. AirQuery forecasts hourly PM2.5 "
        "at 7 monitoring stations, 3 to 48 hours ahead, and gives every forecast an 80% range, not just one number. "
        "The planner page then ranks the coming hours by their **upper** estimate, so a cautious choice is the default."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Stations", len(STATIONS))
    c2.metric("Horizons", "3 to 48 h, every 3 h")
    c3.metric("Interval", "80%")
    c3.caption("From the 10th to the 90th percentile forecast")

    st.header("Problem definition")
    st.markdown(
        "- **Question:** given everything known up to hour *t* at one station, what will hourly PM2.5 be at hour "
        "*t + h*, for *h* = 3, 6, ..., 48, and in what range will it most likely fall?\n"
        "- **One row:** one (station, origin hour *t*, horizon *h*). All stations and horizons are stacked into one "
        "table of 1,719,135 rows with 32 model features.\n"
        "- **Target:** hourly mean PM2.5 (µg/m³) at the target hour, measured by the station. The model predicts the "
        "*change* from the PM2.5 at hour *t*, plus the 10th, 50th and 90th percentiles.\n"
        "- **When a prediction is made:** every hour. Known at that moment: PM2.5 and other pollutant readings up to "
        "and including hour *t*, and a weather forecast for the target hour that was issued at least *h* + 6 hours "
        "before it (the 6 hours is an assumed publication delay).\n"
        "- **Limits of the claim:** forecasts for seven stations, not for the whole city or for any exact place; "
        "one test season; a prediction, not an explanation of what causes PM2.5; not health advice."
    )

    st.header("Data")
    st.markdown(
        "- **Pollution:** CPCB continuous monitoring data for 7 Delhi stations, January 2024 to December 2025, from "
        "data.opencity.in (15-minute readings averaged to hours; negative values and the 999 error code removed).\n"
        "- **Weather:** Open-Meteo *Previous Runs* forecasts, i.e. the forecast that was issued 24, 48 or 72 hours "
        "before each hour. Training therefore sees weather with the same kind of error it will have at prediction "
        "time. Reanalysis weather would be cleaner than any real forecast and would flatter the model.\n"
        "- **Publication delay:** a forecast is only used if it was issued at least *h* + 6 hours before the target "
        "hour. The 6 hours is an assumption, not a measured delay."
    )
    rows = []
    for name, (lat, lon) in STATIONS.items():
        cov, gap = DQ[name]
        rows.append({"Station": name, "PM2.5 coverage (%)": cov, "Longest PM2.5 gap (h)": gap,
                     "Latitude": lat, "Longitude": lon})
    left, right = st.columns([5, 3])
    with left:
        table(pd.DataFrame(rows))
        st.caption("Sirifort has a 2,320-hour PM2.5 gap; ITO has no wind or humidity sensor, so all stations use "
                   "forecast weather. Coordinates come from a published table and have not been checked against CPCB.")
    with right:
        st.map(pd.DataFrame(rows).rename(columns={"Latitude": "lat", "Longitude": "lon"})[["lat", "lon"]])

    st.subheader("Data at a glance")
    st.write("About 17,500 hourly rows per station (January 2024 to December 2025), seven stations. "
             "Delhi PM2.5 is strongly seasonal, so one test season is a narrow check.")
    mon = read_csv("pm25_monthly.csv")
    g1, g2 = st.columns(2)
    with g1:
        if mon is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mon["month"], y=mon["max"], mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=mon["month"], y=mon["min"], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=BLUE_FILL, name="Lowest to highest station"))
            fig.add_trace(go.Scatter(x=mon["month"], y=mon["mean"], mode="lines+markers", name="Mean of stations",
                                     line=dict(color=BLUE, width=2)))
            show(style_fig(fig, "Monthly mean PM2.5 (µg/m³)", height=320))
        else:
            st.info("Monthly summary file not found (pm25_monthly.csv).")
    with g2:
        miss = pd.DataFrame({"Station": [short(k) for k in DQ], "Missing PM2.5 hours (%)": [round(100 - v[0], 1) for v in DQ.values()]})
        fig = go.Figure(go.Bar(x=miss["Station"], y=miss["Missing PM2.5 hours (%)"], marker_color=BLUE))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Missing PM2.5 hours (%)")
        show(fig)
    st.caption("Left: monthly mean over the seven stations, with the range across stations. "
               "Right: share of hours with no valid PM2.5 reading (Sirifort has one long gap).")

    st.subheader("Preprocessing")
    st.markdown(
        "- **Cleaning:** negative readings and the 999 error code are set to missing; 15-minute readings are averaged "
        "to hours; wind direction is averaged as sine and cosine, not as raw degrees. Hours with no reading stay "
        "missing, they are not filled in.\n"
        "- **No learned preprocessing:** nothing is scaled or imputed (LightGBM accepts missing values as they are), "
        "so there is nothing that could be fitted on test data. A row is dropped only if PM2.5 at the origin hour "
        "or the target is missing.\n"
        "- **Features use only the past:** PM2.5 lags and rolling means up to the origin hour; the weather for the "
        "target hour comes from a forecast, not from what was later measured.\n"
        "- **Choices made on validation only:** dropping day-of-year and day-of-week features, the target "
        "definition, model settings and the calibration window were all decided on the three validation folds. "
        "The test window was scored once afterwards.\n"
        "- **Final model:** trained on targets from late January 2024 to September 2025, tested on October to "
        "December 2025."
    )

    st.header("How the forecast is made")
    st.markdown(
        "- One **LightGBM** model per quantile (10%, 50%, 90%), trained on one stacked table where *horizon* is a "
        "feature. Inputs: PM2.5 history up to the forecast moment, other pollutants, hour of day, and the forecast "
        "weather for the target hour. The model predicts the *change* from the current PM2.5.\n"
        "- **Rolling calibration:** raw intervals covered too little, so each day the interval edges are widened or "
        "narrowed using the last 14 days of forecasts whose outcome is already known.\n"
        "- **Baselines:** persistence (PM2.5 now) and the same hour the day before. The model has to beat the "
        "better of the two at each horizon."
    )
    st.header("How it was tested")
    st.markdown(
        "- Rolling-origin validation: three folds, each trained on the past only and scored on the next season "
        "(Sep to Dec 2024, Jan to Mar 2025, Jul to Sep 2025). All tuning happened on these folds.\n"
        "- The final test window, **October to December 2025**, was scored once, after every choice was frozen.\n"
        "- Leakage checks: time-based splits only; forecast weather instead of reanalysis; no value from after the "
        "forecast moment in any feature; calibration at each day uses only outcomes known before that day."
    )
    st.header("Why these choices")
    st.markdown(
        "- **Gradient-boosted trees (LightGBM):** they handle missing values, interactions and non-linear effects "
        "of weather without scaling, and train quickly on 1.7 million rows. Only this model family was tried, "
        "so the site makes no claim that it is the best one.\n"
        "- **Quantile objective:** training directly for the 10th, 50th and 90th percentiles gives a range without "
        "assuming a distribution for the errors.\n"
        "- **One model across horizons:** *h* is a feature, so the 16 horizons share what they learn instead of "
        "needing 48 separate models.\n"
        "- **Change from current PM2.5 as the target:** the forecast starts from the latest reading, which is "
        "also what persistence does, so the model has to add something to it.\n"
        "- **Calendar features dropped:** keeping day-of-year and day-of-week together with a level target scored "
        "worse on the validation folds (see Results, Ablations; two changes were made together in that run).\n"
        "- **Calibration:** raw ranges covered too little, so they are re-calibrated every day from recent known outcomes."
    )
    st.header("Reproduce")
    st.markdown(
        "- **Code:** [github.com/tanujverkala/AirQuery](https://github.com/tanujverkala/AirQuery)\n"
        "- **Data:** the seven station files are downloaded by `get_data.py` from data.opencity.in; weather forecasts "
        "by `fetch_weather.py` from the Open-Meteo Previous Runs API. Nothing is stored in the repository except the "
        "small result files this website reads.\n"
        "- **Steps and splits:** `run_pipeline.py` builds the data; `train_model.py`, `calibrate_rolling.py` and "
        "`error_analysis.py` produce the results; `prepare_app_data.py` builds this site's `data/` folder. "
        "The README lists the order, and the split dates are on this page."
    )
    meta = read_meta()
    if meta:
        st.caption(f"Data folder built {meta.get('created', '')}. Test forecasts cover targets from "
                   f"{meta.get('first_target', '')[:10]} to {meta.get('last_target', '')[:10]}.")


def page_explorer(d):
    st.header("Forecast Explorer")
    st.write("Pick a station, a horizon and a stretch of the test period. Every point on the lines was forecast "
             "*h* hours earlier, using only what was known then. Compare the forecast with what happened and with "
             "the two simple baselines.")
    stations = sorted(d["station"].unique())
    hs = sorted(int(h) for h in d["horizon"].unique())
    t0, t1 = d["target_ts"].min().normalize(), d["target_ts"].max().normalize()
    mid = (t0 + (t1 - t0) / 2).date()
    c1, c2, c3, c4 = st.columns([2, 2, 1.4, 1.6])
    stn = c1.selectbox("Station", stations, format_func=short, key="ex_station")
    h = c2.select_slider("Horizon (hours ahead)", options=hs, value=24 if 24 in hs else hs[len(hs) // 2], key="ex_h")
    start = c3.date_input("From", value=mid, min_value=t0.date(), max_value=t1.date(), key="ex_start")
    span = c4.radio("Days shown", [3, 7, 14], index=1, horizontal=True, key="ex_span")
    a = pd.Timestamp(start)
    g = d[(d["station"] == stn) & (d["horizon"] == h) & (d["target_ts"] >= a)
          & (d["target_ts"] < a + pd.Timedelta(days=span))].sort_values("target_ts")
    if g.empty:
        st.info("No forecasts for this station, horizon and period.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["target_ts"], y=g["q90"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["target_ts"], y=g["q10"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=BLUE_FILL, name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["target_ts"], y=g["y"], mode="lines", name="Observed", line=dict(color=GREY, width=2)))
    fig.add_trace(go.Scatter(x=g["target_ts"], y=g["q50"], mode="lines", name="Forecast (median)",
                             line=dict(color=BLUE, width=2.5)))
    base_name = "Persistence (PM2.5 at forecast time)"
    fig.add_trace(go.Scatter(x=g["target_ts"], y=g["persistence"], mode="lines", name=base_name,
                             line=dict(color=ORANGE, width=1.5, dash="dot"), visible="legendonly"))
    fig.add_trace(go.Scatter(x=g["target_ts"], y=g["yesterday"], mode="lines", name="Same hour yesterday",
                             line=dict(color=ORANGE, width=1.5, dash="dash")))
    show(style_fig(fig, "PM2.5 (µg/m³)"))

    mae = lambda c: float((g["y"] - g[c]).abs().mean())
    naive = min(mae("persistence"), mae("yesterday"))
    inside = float(((g["y"] >= g["q10"]) & (g["y"] <= g["q90"])).mean())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast MAE", f"{mae('q50'):.1f} µg/m³", f"{mae('q50') - naive:+.1f} vs best baseline", delta_color="inverse")
    m2.metric("Persistence MAE", f"{mae('persistence'):.1f}")
    m3.metric("Same hour yesterday MAE", f"{mae('yesterday'):.1f}")
    m4.metric("Outcomes inside the 80% range", f"{100 * inside:.0f}%", f"{len(g)} hours")
    st.caption("A gap in the lines means the observation or an input was missing for that hour. Over a few days these "
               "numbers are noisy; the Results page has the full test-window comparison.")
    with st.expander("Show the numbers"):
        t = g[["target_ts", "y", "q10", "q50", "q90", "persistence", "yesterday"]].copy()
        num = [c for c in t.columns if c != "target_ts"]
        t[num] = t[num].round(1)
        t.columns = ["Target hour", "Observed", "10%", "Median", "90%", "Persistence", "Yesterday"]
        table(t)


def page_planner(d):
    st.header("Best Time to Go Out")
    st.write("A replay of the test period: choose a moment, pretend it is *now*, and see what the model would have "
             "told you about the next 48 hours. The model only uses what was known at that moment.")
    stations = sorted(d["station"].unique())
    t0, t1 = d["origin_ts"].min().normalize() + pd.Timedelta(days=1), d["origin_ts"].max().normalize() - pd.Timedelta(days=2)
    mid = (t0 + (t1 - t0) / 2).date()
    c1, c2, c3 = st.columns([2, 1.4, 1.6])
    stn = c1.selectbox("Station", stations, format_func=short, key="pl_station")
    day = c2.date_input("Pretend it is (date)", value=mid, min_value=t0.date(), max_value=t1.date(), key="pl_day")
    hour = c3.select_slider("Hour", options=list(range(24)), value=8, format_func=lambda x: f"{x:02d}:00", key="pl_hour")
    c4, c5, c6 = st.columns([2, 2, 2])
    lo, hi = c4.slider("Hours of the day I would go out", 0, 23, (6, 21), key="pl_window")
    sensitive = c5.toggle("Sensitive group (children, older adults, breathing problems)", key="pl_sens")
    reveal = c6.toggle("Reveal what actually happened", key="pl_reveal")

    now = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    avail = d.loc[d["station"] == stn, "origin_ts"].drop_duplicates().sort_values()
    earlier = avail[avail <= now]
    if earlier.empty:
        st.info("No forecast was issued at or before that moment for this station.")
        return
    origin = earlier.iloc[-1]
    if origin != now:
        st.warning(f"No forecast for exactly {when(now)}; showing the latest one before it, issued {when(origin)}.")
    f = d[(d["station"] == stn) & (d["origin_ts"] == origin)].sort_values("horizon").copy()
    f = f[f["target_ts"].dt.hour.between(lo, hi)]
    if f.empty:
        st.info("None of the forecast hours fall inside your time window.")
        return
    thr = 60 if sensitive else 90
    f["flag"] = f["q90"] > thr
    best = f.nsmallest(3, ["q90", "q50"]).sort_values("target_ts")
    top = best.sort_values(["q90", "q50"]).iloc[0]

    st.subheader(f"Cleanest time to go: {when(top['target_ts'])}")
    st.write(f"Median forecast **{top['q50']:.0f} µg/m³** ({category(top['q50'])}); 80% range "
             f"{top['q10']:.0f} to {top['q90']:.0f}. Ranked by the upper estimate, not the median.")
    if (f["q90"] > thr).all():
        st.warning(f"Every forecast hour in your window has an upper estimate above {thr} µg/m³ "
                   f"({category(thr + 1)} or worse is possible). The ranking only says which hours are least bad.")

    fig = go.Figure()
    fig.add_hline(y=thr, line_dash="dash", line_color=GREY,
                  annotation_text=f"alert level {thr}" + (" (sensitive group)" if sensitive else ""),
                  annotation_position="top left")
    fig.add_trace(go.Scatter(x=f["target_ts"], y=f["q50"], mode="lines+markers", name="Forecast (median) with 80% range",
                             line=dict(color=BLUE, width=1.5), marker=dict(size=7, color=BLUE),
                             error_y=dict(type="data", symmetric=False, array=(f["q90"] - f["q50"]).clip(lower=0),
                                          arrayminus=(f["q50"] - f["q10"]).clip(lower=0), color=BLUE, thickness=1.5, width=3)))
    fig.add_trace(go.Scatter(x=best["target_ts"], y=best["q50"], mode="markers", name="Three cleanest hours",
                             marker=dict(size=15, color=GREEN, symbol="diamond", line=dict(width=2, color="white"))))
    if reveal:
        fig.add_trace(go.Scatter(x=f["target_ts"], y=f["y"], mode="markers", name="What actually happened",
                                 marker=dict(size=9, color=GREY, symbol="x")))
    show(style_fig(fig, "PM2.5 (µg/m³)", height=440))

    out = pd.DataFrame({
        "Time": f["target_ts"].map(when), "Median": f["q50"].round(0), "80% range": f["q10"].round(0).astype(int).astype(str)
        + " to " + f["q90"].round(0).astype(int).astype(str),
        "Category (median)": f["q50"].map(category),
        "Category (upper estimate)": f["q90"].map(category),
        f"Upper estimate above {thr}?": np.where(f["flag"], "yes", "no"),
        "Cleanest 3": np.where(f["target_ts"].isin(best["target_ts"]), "yes", ""),
    })
    if reveal:
        out["Actual"] = f["y"].round(0).values
    table(out)

    bias = weighted_bias()
    cov = rolling_coverage().get("test")
    notes = ["**This is not health advice.** The categories are the CPCB PM2.5 bands (24-hour breakpoints, applied here "
             "to single hours); they are a guide, not a clinical threshold. Follow official health advisories."]
    if bias is not None and bias < -5:
        notes.append(f"**Why the upper estimate:** in the test period, whenever the median forecast was above 60 µg/m³ it was on "
                     f"average {abs(bias):.0f} µg/m³ **too low**. The upper end of the range is the safer guide.")
    if cov is not None:
        notes.append(f"The 80% range contained {100 * cov:.0f}% of outcomes across the whole test window.")
    for n in notes:
        st.markdown("- " + n)


def page_results(d):
    st.header("Results")
    tabs = st.tabs(["Test window", "Validation folds", "Interval calibration", "Ablations", "Where it errs", "Feature importance"])

    with tabs[0]:
        m = read_csv("model_test_metrics.csv")
        if m is None:
            st.info("model_test_metrics.csv not found.")
        else:
            st.write("October to December 2025, scored once. Skill is 1 − MAE(model) / MAE(best baseline) at that horizon; "
                     "0 means no better than the best of persistence and same-hour-yesterday.")
            t = m[["horizon", "mae_model", "mae_best_naive", "skill_vs_best_naive", "coverage80", "mean_width"]].copy()
            t.columns = ["Horizon (h)", "Model MAE", "Best baseline MAE", "Skill vs best baseline",
                         "Raw 80% coverage", "Raw mean width"]
            table(t.round(2))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m["horizon"], y=m["mae_best_naive"], name="Best baseline", mode="lines",
                                     line=dict(color=ORANGE, width=2.5)))
            fig.add_trace(go.Scatter(x=m["horizon"], y=m["mae_model"], name="Model (median)", mode="lines",
                                     line=dict(color=BLUE, width=2.5)))
            fig.update_xaxes(title="Horizon (hours)")
            show(style_fig(fig, "MAE (µg/m³)", height=340))
            st.write(f"Mean skill over horizons: **{100 * (1 - (m['mae_model'] / m['mae_best_naive']).mean()):.1f}%**.")
        rc = read_csv("rolling_check.csv")
        if rc is not None:
            r = rc[rc["fold"] == "test"].copy()
            if len(r):
                st.subheader("Interval coverage on the test window (target 0.80)")
                r = r.rename(columns={"variant": "Method", "coverage": "Coverage", "below10": "Below 10% line",
                                      "above10": "Above 90% line", "width_ratio": "Width vs raw"})
                table(r[["Method", "Coverage", "Below 10% line", "Above 90% line", "Width vs raw"]].round(3))
        ev = read_csv("model_test_events.csv")
        if ev is not None:
            st.subheader("Poor-air alarms (PM2.5 above 90 µg/m³)")
            e = ev.drop(columns="fold").rename(columns={"rule": "Alarm rule", "event_rate": "Share of hours that were events",
                                                         "alarms": "Alarms", "precision": "Precision", "recall": "Recall"})
            table(e.round(3))
            st.caption("The 90 µg/m³ line is the start of the CPCB 'Moderately polluted' to 'Poor' boundary (91+), taken from a "
                       "secondary source and applied to hourly values. 'model_q90' raises an alarm when the upper estimate "
                       "is above the line.")

    with tabs[1]:
        v = read_csv("model_val_metrics.csv")
        if v is None:
            st.info("model_val_metrics.csv not found.")
        else:
            st.write("Skill vs the best baseline on the three validation folds the model was tuned on.")
            p = v.pivot(index="horizon", columns="fold", values="skill_vs_best_naive").reset_index()
            fig = go.Figure()
            for col, color in zip([c for c in p.columns if c != "horizon"], [BLUE, ORANGE, GREEN]):
                fig.add_trace(go.Scatter(x=p["horizon"], y=p[col], name=col, mode="lines", line=dict(color=color, width=2.5)))
            fig.add_hline(y=0, line_color=GREY)
            fig.update_xaxes(title="Horizon (hours)")
            show(style_fig(fig, "Skill vs best baseline", height=340))
            table(p.round(3).rename(columns={"horizon": "Horizon (h)"}))
            st.caption("Fold 1 (Sep to Dec 2024) trains without a smog season; fold 2 (Jan to Mar 2025) trains with one; "
                       "fold 3 (Jul to Sep 2025) is the monsoon.")

    with tabs[2]:
        rc = read_csv("rolling_check.csv")
        if rc is None:
            st.info("rolling_check.csv not found.")
        else:
            st.write("Raw intervals covered less than 80%. Rolling calibration widens or narrows each side using only "
                     "forecasts whose outcome was already known. A static correction learned from earlier folds is "
                     "shown for comparison.")
            table(rc.round(3).rename(columns={"fold": "Set", "variant": "Method", "coverage": "Coverage",
                                              "below10": "Below 10% line", "above10": "Above 90% line",
                                              "width_ratio": "Width vs raw", "calibrated_share": "Share corrected"}))

    with tabs[3]:
        ab = read_csv("ablation.csv")
        if ab is None:
            st.info("ablation.csv not found.")
        else:
            st.write("Mean over horizons of MAE(model) / MAE(best baseline) on each validation fold. Lower is better; "
                     "above 1 means worse than the baseline. One run each, one random seed.")
            table(ab.round(3).rename(columns={"variant": "Variant", "fold1": "Fold 1", "fold2": "Fold 2",
                                              "fold3": "Fold 3", "mean": "Mean"}))

    with tabs[4]:
        for tag, label in [("test_error", "Test window"), ("val_error", "Validation folds")]:
            by_st = read_csv(f"{tag}/by_station.csv")
            if by_st is None:
                continue
            st.subheader(label)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**By station**")
                t = by_st[["fold", "station", "skill", "bias", "coverage_cal"]].round(2)
                t.columns = ["Set", "Station", "Skill vs baseline", "Bias (µg/m³)", "Calibrated coverage"]
                table(t)
            with c2:
                fl = read_csv(f"{tag}/by_forecast_level.csv")
                if fl is not None:
                    st.markdown("**By forecast level** (bias conditioned on what the model predicted)")
                    t = fl[["fold", "fband", "n", "mean_pm25", "bias", "coverage_cal"]].round(2)
                    t.columns = ["Set", "Forecast band", "Rows", "Mean observed", "Bias (µg/m³)", "Calibrated coverage"]
                    table(t)
            wd = read_csv(f"{tag}/worst_days.csv")
            if wd is not None:
                st.markdown("**Worst days by mean absolute error**")
                table(wd.round(1).rename(columns={"fold": "Set", "target_date": "Date", "n": "Rows",
                                                  "mean_pm25": "Mean PM2.5", "mae_model": "Model MAE",
                                                  "mae_naive": "Baseline MAE", "bias": "Bias"}))
            figs = sorted((DATA / "figures" / tag.split("_")[0]).glob("*.png"))
            for p in figs:
                image(p)

    with tabs[5]:
        fi = read_csv("final_feature_importance.csv")
        if fi is None:
            fi = read_csv("feature_importance.csv")
        if fi is None:
            st.info("No feature importance file found.")
        else:
            top = fi.sort_values("gain_pct", ascending=False).head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["gain_pct"], y=top["feature"], orientation="h", marker_color=BLUE))
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Share of total gain (%)")
            show(fig)
            st.caption("Gain importance describes how the trained trees split, not what causes PM2.5. 'wx_' features are "
                       "forecast weather for the target hour.")


def page_conclusions(d):
    st.header("Conclusions and Limitations")
    m = read_csv("model_test_metrics.csv")
    ab = read_csv("ablation.csv")
    st.subheader("What the model does")
    if ab is not None:
        f = ab.iloc[0]
        st.markdown(f"- On the validation folds, the model's error was on average **{100 * (1 - f['fold1']):.0f}%, "
                    f"{100 * (1 - f['fold2']):.0f}% and {100 * (1 - f['fold3']):.0f}% lower** than the better of "
                    "persistence and same-hour-yesterday (folds 1, 2, 3).")
    if m is not None:
        sk = 100 * (1 - (m["mae_model"] / m["mae_best_naive"]).mean())
        lo, hi = 100 * m["skill_vs_best_naive"].min(), 100 * m["skill_vs_best_naive"].max()
        verdict = ("In this season the model is about as good as the simple baselines, not better." if sk < 5
                   else "In this season the model beat the simple baselines.")
        st.markdown(f"- On the final test window (Oct to Dec 2025, mean PM2.5 {d['y'].mean():.0f} µg/m³) the gain over the "
                    f"best baseline was **{sk:.1f}%** on average ({lo:+.0f}% to {hi:+.0f}% depending on the horizon). {verdict}")
    cov = rolling_coverage()
    if cov:
        parts = ", ".join(f"{k} {100 * v:.0f}%" for k, v in cov.items())
        st.markdown(f"- The part that adds value beyond the baselines is the **uncertainty**: the calibrated 80% range "
                    f"contained this share of outcomes ({parts}). The baselines give no range at all.")

    st.subheader("Limitations")
    bs, wb = read_csv("test_error/by_station.csv"), weighted_bias()
    if bs is not None and wb is not None:
        st.markdown(f"- **The median forecast ran low in the test window:** below the observed value on average at "
                    f"{int((bs['bias'] < 0).sum())} of {len(bs)} stations (station averages {bs['bias'].min():.0f} to "
                    f"{bs['bias'].max():+.0f} µg/m³), and by {abs(wb):.0f} µg/m³ on average whenever it predicted above 60. "
                    "The planner therefore ranks by the upper estimate.")
    st.markdown(
        "- **Two winters only.** Test performance is one season. Both validation fold 1 (trained without a smog season) "
        "and the test window (trained with one) show little or no gain in heavy smog; with two winters the cause cannot be "
        "separated. More years would help most.\n"
        "- **Seven correlated stations in one city.** They share weather and sources, so they are not seven independent tests.\n"
        "- **Data gaps.** Sirifort has a 2,320-hour PM2.5 gap; weather forecasts in the archive start in mid January 2024.\n"
        "- **Forecast weather has its own errors**, and the provider's underlying forecast model can change over time; both "
        "pass through into this model.\n"
        "- **Assumptions:** a 6-hour publication delay for forecast weather; CPCB bands applied to hourly values; "
        "a single random seed and one run per ablation.\n"
        "- **Calibration** was tuned (window length) on the same validation folds, and does not reach the 80% target exactly.\n"
        "- **Not health advice.** The planner is a demonstration of forecasting with uncertainty."
    )
    st.subheader("Licensing")
    st.markdown("- Open-Meteo's free tier is for non-commercial use. CPCB data terms of use: not yet verified. "
                "Check both before any commercial or public production use.")
    st.subheader("What would come next")
    st.markdown("- More years of data and a smog season in every training window.\n"
                "- A rolling correction of the median itself (the same idea as the interval calibration). It was not tried, "
                "because the test window has already been scored once; any such change would need to be reported as made after the test.\n"
                "- A live data feed instead of a replay, and per-station calibration.")
    meta = read_meta()
    if meta:
        st.caption(f"Text written from the frozen run; data folder built {meta.get('created', '')}.")


# ------------------------------------------------------------------ main
d = read_preds()
page = st.sidebar.radio("Page", PAGES, key="page")
st.sidebar.markdown("---")
st.sidebar.caption("Replay of Oct to Dec 2025 forecasts. Not health advice.")

if d is None:
    st.error("data/test_predictions.parquet not found. Run prepare_app_data.py and copy the data folder next to app.py.")
    st.stop()

{"Problem and Data": page_problem, "Forecast Explorer": page_explorer, "Best Time to Go Out": page_planner,
 "Results": page_results, "Conclusions and Limitations": page_conclusions}[page](d)
