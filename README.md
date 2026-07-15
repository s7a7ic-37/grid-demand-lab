# Grid Demand Lab

A half-hourly Great Britain electricity demand forecast, replayed against held-out historical days. LightGBM is compared with simple same-time yesterday and last week baselines on the final four weeks. The replay uses the model trained only before that holdout.

Data: NESO Historic Demand Data 2025 (`ND`, MW), provided under the NESO Open Data Licence. NESO may revise published values.

## Run

Requires Python 3.11 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

`pytest` runs checks for time-safe features, duplicate input rows and missing history. `train.py` writes the model and holdout metrics under `models/`.
