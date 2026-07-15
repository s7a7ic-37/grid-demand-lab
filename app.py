from pathlib import Path
import json

import pandas as pd
import streamlit as st
from lightgbm import Booster

from forecast import forecast_day, load_demand

ROOT = Path(__file__).parent
st.set_page_config(page_title="Grid Demand Lab", page_icon="⚡", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_demand(ROOT / "data" / "demand_2025.csv")


@st.cache_resource
def get_model() -> Booster:
    return Booster(model_file=str(ROOT / "models" / "demand_model.txt"))


@st.cache_data
def get_metrics() -> dict:
    return json.loads((ROOT / "models" / "metrics.json").read_text())


data = get_data()
model = get_model()
metrics = get_metrics()
dates = sorted(data["date"].dt.strftime("%Y-%m-%d").unique())

st.title("Grid Demand Lab")
st.caption("A small forecasting workbench for Great Britain’s half-hourly electricity demand.")
choice = st.selectbox("Replay a day", dates[-28:], index=len(dates[-28:]) - 1)
actual = data[data["date"].eq(pd.Timestamp(choice))][["period", "demand_mw"]]
predicted = forecast_day(data, model, choice)
result = actual.merge(predicted, on="period")

left, right = st.columns(2)
left.metric("Forecast mean", f"{result['forecast_mw'].mean():,.0f} MW")
right.metric("Actual mean", f"{result['demand_mw'].mean():,.0f} MW")
st.line_chart(result.set_index("period")[["demand_mw", "forecast_mw"]])
st.metric("Holdout MAE · final 28 days", f"{metrics['model_mae']:,.0f} MW",
          f"{(1 - metrics['model_mae'] / metrics['week_mae']) * 100:.0f}% lower than last week")
st.caption(f"{metrics['holdout_start']} to {metrics['holdout_end']}. LightGBM uses previous-day and previous-week demand with calendar features.")
st.caption("Replay only: this view compares historical forecasts with known demand. It is not a live system forecast.")
st.link_button("NESO data and licence", "https://www.neso.energy/data-portal/historic-demand-data")
