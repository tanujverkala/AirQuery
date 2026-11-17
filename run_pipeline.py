"""Run the AirQuery data pipeline, skipping every step whose output already exists.

Steps (in order): get_data, clean_data, fetch_weather, build_features.
All paths are relative to the folder this file sits in, so run it from anywhere:
    python /content/drive/MyDrive/airsafe/run_pipeline.py
    python run_pipeline.py --force build_features      # redo one step (or several)
A step counts as done when its final output file exists.
"""
import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def have(*patterns):
    return lambda: any(list(HERE.glob(p)) for p in patterns)


STEPS = [
    ("get_data",       "get_data.py",       lambda: len(list((HERE / "data/raw").glob("*.csv"))) >= 7),
    ("clean_data",     "clean_data.py",     have("data/clean/hourly_all.csv")),
    ("fetch_weather",  "fetch_weather.py",  have("data/weather/weather_leads.csv")),
    ("build_features", "build_features.py", have("data/features/features.parquet",
                                                 "data/features/features.csv.gz")),
]


def run(script):
    p = subprocess.Popen([sys.executable, str(HERE / script)], cwd=HERE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:  # stream so the notebook shows progress
        print("   ", line.rstrip(), flush=True)
    return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[], choices=[s[0] for s in STEPS])
    args, _ = ap.parse_known_args()  # ignore Colab's injected "-f kernel.json"
    for name, script, done in STEPS:
        if name not in args.force and done():
            print(f"[skip] {name}: output already exists")
            continue
        print(f"[run ] {name}")
        t0 = time.time()
        code = run(script)
        if code != 0:
            sys.exit(f"[fail] {name} exited with code {code}. Fix it and re-run; "
                     "finished steps will be skipped.")
        print(f"[done] {name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
