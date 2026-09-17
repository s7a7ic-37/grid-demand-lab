from pathlib import Path
import json

import pandas as pd
import streamlit as st
from lightgbm import Booster

from forecast import fetch_live_demand, forecast_day

ROOT = Path(__file__).parent
st.set_page_config(page_title="Grid Demand Lab", page_icon="⚡", layout="wide")


@st.cache_data(ttl=21600)
def get_live_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_live_demand()


@st.cache_resource
def get_model() -> Booster:
    return Booster(model_file=str(ROOT / "models" / "demand_model.txt"))


@st.cache_data
def get_metrics() -> dict:
    return json.loads((ROOT / "models" / "metrics.json").read_text())


if st.button("Refresh live data"):
    get_live_data.clear()
    st.rerun()

with st.spinner("Loading the latest NESO demand data…"):
    try:
        actual_data, neso_forecast = get_live_data()
    except Exception:
        st.error("NESO’s live feed could not be loaded. Try refreshing in a moment.")
        st.stop()
model = get_model()
metrics = get_metrics()
latest_date = actual_data["date"].max().normalize()
target_date = latest_date + pd.DateOffset(days=1)

st.title("Grid Demand Lab")
st.caption("A daily view of Great Britain’s half-hourly electricity demand, with an ML forecast beside NESO’s forecast.")
st.caption(f"Latest outturn in NESO’s feed: {latest_date:%d %b %Y}. Their published data can arrive several days after settlement.")
predicted = forecast_day(actual_data, model, target_date.strftime("%Y-%m-%d"))
official = neso_forecast[neso_forecast["date"].eq(target_date)][["period", "demand_mw"]]
result = predicted.merge(official, on="period", how="left", suffixes=("", "_neso"))
result = result.rename(columns={"demand_mw": "neso_forecast_mw"})

st.metric(f"Average demand · {target_date:%d %b}", f"{result['forecast_mw'].mean():,.0f} MW")
neso_count = int(result["neso_forecast_mw"].notna().sum())
if neso_count:
    st.metric("NESO forecast · same day", f"{result['neso_forecast_mw'].mean():,.0f} MW")
    if neso_count < 48:
        st.caption(f"NESO has published {neso_count} of 48 half-hour forecasts for this day.")
else:
    st.caption("NESO’s forecast values are currently missing; showing the model forecast on its own.")
chart = result.set_index("period")[["forecast_mw"]].rename(columns={"forecast_mw": "Model forecast"})
chart.index.name = "Half-hour settlement period"
if neso_count:
    chart["NESO forecast"] = result.set_index("period")["neso_forecast_mw"]
st.line_chart(chart, y_label="Demand (MW)", width="stretch")
fallback_count = int(predicted["used_weekly_fallback"].sum())
if fallback_count:
    st.caption(f"Used last week’s matching periods where today’s feed has not filled {fallback_count} half-hour readings yet.")
st.metric("Historical holdout MAE · final 28 days of 2025", f"{metrics['model_mae']:,.0f} MW",
          f"{(1 - metrics['model_mae'] / metrics['week_mae']) * 100:.0f}% lower than last week")
st.caption(f"2025 holdout: {metrics['holdout_start']} to {metrics['holdout_end']}. The live model uses 2025 history; its inputs are the latest actual outturns.")
st.link_button("NESO data and licence", "https://www.neso.energy/data-portal/historic-demand-data")
