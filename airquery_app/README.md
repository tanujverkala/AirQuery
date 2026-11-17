# AirQuery: 48-hour PM2.5 forecasts with uncertainty for Delhi

Forecasts hourly PM2.5 at 7 Delhi CPCB stations, 3 to 48 hours ahead, with an 80% range (LightGBM quantile models, rolling calibration), plus a "best time to go out" planner that replays Oct to Dec 2025. The website reads only precomputed files from `data/`.

Live site: `[Streamlit URL]`
Code: https://github.com/tanujverkala/AirQuery

## Repository layout
- Root: pipeline scripts (`get_data.py`, `clean_data.py`, `fetch_weather.py`, `build_features.py`, `baselines.py`, `train_model.py`, `calibrate.py`, `calibrate_rolling.py`, `error_analysis.py`, `prepare_app_data.py`, `run_pipeline.py`) and `AirQuery_reproduce.ipynb` (the run order).
- `airquery_app/`: `app.py`, `requirements.txt`, `data/` (small result files the site reads).
- Not stored here: raw data, weather files, cached models and large result files. They are re-created by the steps below.

## Data access
- PM2.5 and other pollutants: seven station files from data.opencity.in (CPCB / DPCC, Jan 2024 to Dec 2025), downloaded by `get_data.py`.
- Weather: Open-Meteo Previous Runs API, downloaded by `fetch_weather.py`. No API key is used.
- Terms of use: Open-Meteo's free tier is for non-commercial use. CPCB data terms: `[not yet verified]`.

## Splits (time-based, no shuffling)
- Rolling-origin validation: fold 1 trains Jan to Aug 2024, validates Sep to Dec 2024; fold 2 trains 2024, validates Jan to Mar 2025; fold 3 trains to Jun 2025, validates Jul to Sep 2025.
- Final model: trained on targets from 22 Jan 2024 to Sep 2025. Test window: Oct to Dec 2025, scored once.

## Reproduce the results (Google Colab or any machine with Python 3)
Open `AirQuery_reproduce.ipynb`, or run in order from the repository root:

    pip install lightgbm pandas numpy pyarrow scikit-learn requests matplotlib
    python run_pipeline.py                      # download, clean, weather, features
    python baselines.py
    python train_model.py --configs A,B         # validation, picks the config
    python calibrate.py
    python calibrate_rolling.py
    python error_analysis.py
    python train_model.py --final-test          # use once
    python calibrate_rolling.py --test
    python error_analysis.py --test
    python prepare_app_data.py                  # builds airquery_app/data

The full run takes a long time. Results can differ slightly between machines and library versions. The ablation runs on the Results page (log-ratio target, no weather, calendar features kept) used extra flags of `train_model.py` (`--target`, `--drop`, `--tag`); see its docstring. Ablation numbers were copied by hand into `prepare_app_data.py`.

## Run the website locally
    cd airquery_app
    pip install -r requirements.txt
    streamlit run app.py
Tested with streamlit 1.66.1, plotly 7.1.0, pandas 3.0.5.

## Deploy
Streamlit Community Cloud: repository `tanujverkala/AirQuery`, main file `airquery_app/app.py`.

## Notes
- Station coordinates on the site come from a published table and are not checked against CPCB.
- The 6-hour weather publication delay is an assumption.
- Not health advice.
