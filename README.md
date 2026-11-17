# AirQuery: 48-hour PM2.5 forecasting with uncertainty for Delhi air quality stations

AirQuery forecasts hourly PM2.5 at 7 Delhi monitoring stations, 3 to 48 hours ahead (every 3 hours), and gives every forecast an 80% range (10th to 90th percentile). A "best time to go out" planner replays the October to December 2025 test period and ranks the next hours by their upper estimate.

- Live website (GitHub Pages): \[https\://tanujverkala.github.io/AirQuery/\]  
- Full app (Streamlit Community Cloud): \[https\://airquery-n349ezf53j3qg8w6gwsoij.streamlit.app/\]  
- Source code: [https\://github.com/tanujverkala/AirQuery](https://github.com/tanujverkala/AirQuery)  
- Course: ES 335 (individual project)

## Question and scope

Given everything known up to hour t at one station, what will hourly PM2.5 be at hour t \+ h, for h \= 3, 6, ..., 48, and in what range will it most likely fall? One row of the model table is one (station, origin hour, horizon). The target is the hourly mean PM2.5 (µg/m³) at the target hour. Forecasts are for the seven stations only, not for the whole city. The result is a prediction, not an explanation of what causes PM2.5, and it is not health advice.

## Repository layout

- Root: pipeline scripts and the notebook that runs them  
  - `get_data.py`, `clean_data.py`, `fetch_weather.py`, `build_features.py`, `baselines.py`  
  - `train_model.py`, `calibrate.py`, `calibrate_rolling.py`, `error_analysis.py`  
  - `prepare_app_data.py`, `run_pipeline.py`  
  - `AirQuery_reproduce.ipynb` (run order)  
- `airquery_app/`: the Streamlit website (`app.py`, `requirements.txt`) and `data/`, the small result files the site reads.  
- `docs/index.html`: the GitHub Pages page that embeds and links to the Streamlit app.  
- `models/`: saved LightGBM models of the final run.  
- Not stored in the repository: raw data, weather files and large intermediate result files. The steps below re-create them.

## Data access

- PM2.5 and other pollutants: seven station files (CPCB / DPCC, January 2024 to December 2025\) from data.opencity.in, downloaded by `get_data.py`.  
- Weather: Open-Meteo Previous Runs API (forecasts issued 1, 2 and 3 days earlier), downloaded by `fetch_weather.py`. No API key is used.  
- If the automatic download is refused (HTTP 403), `get_data.py` prints the steps for a manual download: save the seven files into `data/raw/` under the names listed in `get_data.py` and run `run_pipeline.py` again. Finished steps are skipped.  
- Terms of use: Open-Meteo's free tier is for non-commercial use. CPCB data terms of use: \[not yet verified\].  
- Data checks: negative readings and the 999 error code are set to missing, 15-minute readings are averaged to hours, and missing hours stay missing. Sirifort has a long PM2.5 gap (about 2,320 hours); the Data section of the website shows coverage per station.

## Method in short

- Baselines: persistence (PM2.5 at the forecast time) and the same hour the day before. The model has to beat the better of the two at each horizon.  
- Model: LightGBM, one model each for the 10th, 50th and 90th percentile, trained on one stacked table with the horizon as a feature. It predicts the change from the current PM2.5. Inputs are PM2.5 history up to the forecast hour, other pollutants and forecast weather for the target hour (issued at least h \+ 6 hours earlier; the 6 hours is an assumption).  
- Preprocessing: no scaling or imputation, so nothing is learned from test data. Calendar features (day of year, day of week) were dropped after validation.  
- Interval calibration: each day the interval edges are adjusted using the last 14 days of forecasts whose outcome is already known.

## Splits (time-based, no shuffling)

- Validation, rolling origin: fold 1 trains Jan to Aug 2024 and validates Sep to Dec 2024; fold 2 trains 2024 and validates Jan to Mar 2025; fold 3 trains to Jun 2025 and validates Jul to Sep 2025\. All choices were made on these folds.  
- Final model: trained on targets from 22 January 2024 to September 2025\.  
- Test: October to December 2025, scored once, after every choice was frozen.

## Reproduce the results

Open `AirQuery_reproduce.ipynb` in Google Colab, or run from the repository root:

```
pip install lightgbm pandas numpy pyarrow scikit-learn requests matplotlib
python run_pipeline.py                      # download, clean, weather, features
python baselines.py
python train_model.py --configs A,B         # validation, picks the configuration
python calibrate.py
python calibrate_rolling.py
python error_analysis.py
python train_model.py --final-test          # run once
python calibrate_rolling.py --test
python error_analysis.py --test
python prepare_app_data.py                  # builds airquery_app/data
```

The full run takes a long time. Results can differ slightly between machines and library versions (one random seed, one run per variant). The ablation rows on the website's Results page (log-ratio target, no weather, calendar features kept) come from extra runs of `train_model.py` with `--target`, `--drop` and `--tag`; their numbers were copied by hand into `prepare_app_data.py`. Its docstring and the notebook list the exact commands.

## Run the website locally

```
cd airquery_app
pip install -r requirements.txt
streamlit run app.py
```

Tested with Streamlit 1.66.1, Plotly 7.1.0 and pandas 3.0.5.

## Limitations

- One test season (Oct to Dec 2025\) and two winters of data in total.  
- Seven stations in one city share weather and sources, so they are not seven independent tests.  
- The median forecast tends to run low in heavy smog; the planner therefore ranks hours by the upper estimate.  
- Weather inputs are forecasts and carry their own errors. The 6-hour publication delay is an assumption.  
- CPCB categories are applied to hourly values although they are defined for 24-hour averages.  
- The calibration window was tuned on the same validation folds.  
- Station coordinates: approximate, used only to place map markers on the website. The source table was not recorded and the values are not checked against CPCB.

## AI-use disclosure

I used Claude (Anthropic) as a coding and writing assistant. It helped write the code for data pipeline, feature table, LightGBM quantile model, interval calibration, error analysis, the Streamlit website, the GitHub Pages page, this README and the website text.

I ran all the code myself on Google Colab and checked the printed outputs. The following are the decisions I made or approved: The project question and scope; using a LightGBM quantile model; choosing rolling calibration over static calibration; testing a log-ratio target and a no-weather variant as ablations; and what to deploy and where. Result numbers on the website come from the output files of the code; the ablation numbers were copied by hand from the printed outputs. AI output was not used for: the data itself, or any of the measured results (these come from running the code). 

## Citations and external material

- Data: PM2.5 and other pollutant readings for seven Delhi stations (CPCB / DPCC), \[[data-opencity](https://data.opencity.in)\], accessed October 2026\.

- Weather: Zippenfenig, P. (2023). Open-Meteo.com Weather API (computer software). [Zenodo](https://doi.org/10.5281/ZENODO.7970649). Previous Runs API,\[[open-meteo](https://open-meteo.com)\], accessed October 2026\. Free API, used for non-commercial purposes.

- Model: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems (NIPS) 30\.

- Software: Python, pandas, NumPy, scikit-learn, Matplotlib, Plotly, Streamlit. Website library versions are in airquery\_app/requirements.txt.

- CPCB PM2.5 categories: secondary source, \[[ies.gov.in, National Air Quality Index](https://ies.gov.in/arthapedia/concept/national-air-quality-index)\] (accessed November 2026), applied to hourly values; not an official CPCB hourly index.

- Station coordinates: approximate, used only to place map markers on the website. The source table was not recorded and the values are not checked against CPCB.

- Interval calibration: split-conformal idea from course material. 
