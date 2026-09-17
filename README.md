# Grid Demand Lab

A live half-hourly Great Britain electricity demand view. It compares a LightGBM next-day forecast against NESO's published forecast. Historical model scores use a separate time-based holdout on 2025 data.

Data: NESO's live Demand Data Update feed for the latest outturns and operational forecasts, plus Historic Demand Data 2025 (`ND`, MW) for training and holdout evaluation. Both are provided under the NESO Open Data Licence. Outturns arrive after settlement and can be revised.

## Run

Requires Python 3.11 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The app fetches the public NESO API at startup and caches it for six hours; use **Refresh live data** to fetch sooner. A partial current day uses the matching periods from last week until today's readings arrive. `pytest` checks time-based features, live feed parsing, duplicate rows and missing history. `train.py` writes the model and holdout metrics under `models/`.
