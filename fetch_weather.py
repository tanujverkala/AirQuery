"""Fetch forecast weather at fixed lead times for the 7 Delhi CPCB stations.

Source: Open-Meteo Previous Runs API. <variable>_previous_dayN is the value that
was forecast N*24 hours before the valid time, so training sees weather with the
same kind of error it will have at prediction time.

Usage:
    python fetch_weather.py --test   # one station, 2 days, prints raw results
    python fetch_weather.py          # full download (cached) + builds the CSV

Output: data/weather/weather_leads.csv  (one row per station-hour, local time)
Raw responses are cached in data/weather_raw/, so an interrupted run resumes.
No API key is needed on the free non-commercial tier.
"""
import argparse
import json
import pathlib
import re
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
START, END = date(2024, 1, 1), date(2025, 12, 31)
WINDOW_DAYS = 14   # keep each call at <=14 days and <=10 variables (lighter rate-limit weight)
LEADS = [1, 2, 3]  # previous_day1..3 = forecast issued 24 / 48 / 72 h before valid time
VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_direction_10m", "precipitation"]
TZ = "Asia/Kolkata"  # CPCB timestamps are local time labelled +0000 (your hourly-profile check)
PAUSE_S = 0.3

# Coordinates from Miinalainen et al. 2022 supplement (3 decimals); sanity-check on a map.
STATIONS = {
    "DTU, Delhi - CPCB":                   (28.749, 77.120),
    "IHBAS, Dilshad Garden, Delhi - CPCB": (28.681, 77.303),
    "ITO, Delhi - CPCB":                   (28.629, 77.241),
    "NSIT Dwarka, Delhi - CPCB":           (28.609, 77.033),
    "Shadipur, Delhi - CPCB":              (28.651, 77.147),
    "Sirifort, Delhi - CPCB":              (28.550, 77.216),
    "Wazirpur, Delhi - DPCC":              (28.700, 77.165),
}

RAW = pathlib.Path("data/weather_raw")
OUT = pathlib.Path("data/weather")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def windows():
    s = START
    while s <= END:
        e = min(s + timedelta(days=WINDOW_DAYS - 1), END)
        yield s, e
        s = e + timedelta(days=1)


def fetch(lat, lon, lead, s, e):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(f"{v}_previous_day{lead}" for v in VARS),
        "start_date": s.isoformat(),
        "end_date": e.isoformat(),
        "timezone": TZ,
    }
    for attempt in range(6):
        try:
            r = requests.get(URL, params=params, timeout=90)
        except requests.exceptions.RequestException as ex:
            wait = 10 * (attempt + 1)
            print(f"  network error ({type(ex).__name__}), retrying in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            sys.exit(f"HTTP {r.status_code} for {r.url}\n{r.text[:500]}")
        return r.json()
    sys.exit("Failed after 6 attempts. Re-run to resume from the cache.")


def test():
    name = next(iter(STATIONS))
    lat, lon = STATIONS[name]
    print("Station:", name)
    for lead in LEADS:
        js = fetch(lat, lon, lead, date(2025, 1, 1), date(2025, 1, 2))
        df = pd.DataFrame(js["hourly"])
        print(f"\n--- lead {lead} ---")
        print("top-level keys:", sorted(js.keys()))
        print("units:", js.get("hourly_units"))
        print("rows:", len(df))
        print(df.head(6).to_string(index=False))
        print("missing %:", (df.isna().mean() * 100).round(1).to_dict())


def download():
    for name, (lat, lon) in STATIONS.items():
        d = RAW / slug(name)
        d.mkdir(parents=True, exist_ok=True)
        for lead in LEADS:
            for s, e in windows():
                f = d / f"lead{lead}_{s.isoformat()}.json"
                if f.exists():
                    continue
                f.write_text(json.dumps(fetch(lat, lon, lead, s, e)))
                time.sleep(PAUSE_S)
        print("done:", name)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in STATIONS:
        per_lead = []
        for lead in LEADS:
            files = sorted((RAW / slug(name)).glob(f"lead{lead}_*.json"))
            df = pd.concat([pd.DataFrame(json.loads(f.read_text())["hourly"]) for f in files],
                           ignore_index=True)
            df = df.rename(columns={f"{v}_previous_day{lead}": f"{v}_lead{lead}" for v in VARS})
            per_lead.append(df.set_index("time"))
        m = pd.concat(per_lead, axis=1).reset_index().rename(columns={"time": "ts"})
        m.insert(0, "station", name)
        frames.append(m)
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"])
    out.to_csv(OUT / "weather_leads.csv", index=False)
    print("rows per station:")
    print(out.groupby("station").size().to_string())
    print("missing % per column:")
    print((out.drop(columns=["station", "ts"]).isna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args, _ = ap.parse_known_args()  # ignore Colab's injected "-f kernel.json"
    if args.test:
        test()
    else:
        download()
        build()
