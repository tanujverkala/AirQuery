# AirQuery: 48-hour PM2.5 forecasts with uncertainty for Delhi

Streamlit website. It reads only precomputed files from `data/` (no model is run on the server).

## 1. Build the data folder (in Colab, in your project folder)
Run first (if not already done):
    !python error_analysis.py
    !python error_analysis.py --test
Then:
    !python prepare_app_data.py
This creates `airquery_app/data/`. Download that folder and put it next to `app.py`.

## 2. Run locally
    pip install -r requirements.txt
    streamlit run app.py
Tested with streamlit 1.66.1, plotly 7.1.0, pandas 3.0.5 (on synthetic data; not yet on your real files).

## 3. Deploy (Streamlit Community Cloud)
Push `app.py`, `requirements.txt`, `README.md` and `data/` to a GitHub repo, then create the app at
share.streamlit.io, choosing the repo and `app.py` as the main file. Verify the current steps on that site.

## Notes
- Unverified: station coordinates, CPCB data terms of use. The 6-hour weather publication delay is an assumption.
- Open-Meteo's free tier is for non-commercial use.
- Not health advice.
