# Grid Demand Lab

A live half-hourly Great Britain electricity demand view. It compares a LightGBM next-day forecast against NESO's published forecast. Historical model scores use a separate time-based holdout on 2025 data.

Data: NESO's live Demand Data Update feed for the latest outturns and operational forecasts, plus Historic Demand Data 2025 (`ND`, MW) for training and holdout evaluation. Both are provided under the NESO Open Data Licence. Outturns arrive after settlement and can be revised.

## Run

Requires Python 3.12 or newer; deployment uses Python 3.13.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
python train.py
streamlit run app.py
```

The app fetches the public NESO API at startup and caches it for six hours; use **Refresh live data** to fetch sooner. A partial current day uses the matching periods from last week until today's readings arrive. `pytest` checks time-based features, live feed parsing, duplicate rows and missing history. `train.py` writes the model and holdout metrics under `models/`.

Tap or hover in the plot to select a half-hour. The UK clock time and forecast values remain above the chart after lifting a finger, including an explicit label when NESO's value is unavailable.

Features match the same calendar date and settlement period one day and seven days earlier, rather than assuming every day has 48 rows. UK clock labels cover 46/48/50-period days. Extra autumn clock-change periods currently reuse period 48 for lag inputs; this is an exploratory forecast, not an operational scheduling model.

The reported MAE belongs to a model trained before the final 28-day holdout. The deployed model is then refitted on all available 2025 history. Refresh updates inputs, not the trained model; no live accuracy claim is made.

The historical backtest table adds a separate 28-day test at each completed quarter-end. Each fold fits the same model settings on earlier data and compares model, previous-day and previous-week MAE on identical periods. The CSV includes the training cutoff and scored period count. These are tests on cleaned historical readings, not a reconstruction of delayed or revised live feeds; no fold is selected to replace the live model.
