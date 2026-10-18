from pathlib import Path
import json

import altair as alt
import pandas as pd
import streamlit as st
from lightgbm import Booster

from forecast import fetch_live_demand, forecast_day, settlement_clock_times

ROOT = Path(__file__).parent
st.set_page_config(page_title="Grid Demand Lab", page_icon="⚡", layout="wide")


@st.cache_data(ttl=21600, show_spinner=False)
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

with st.status("Fetching live half-hourly demand…", expanded=True) as load_status:
    st.write("Loading the latest actual readings and NESO forecast.")
    try:
        actual_data, neso_forecast = get_live_data()
    except Exception:
        load_status.update(label="NESO data is unavailable", state="error")
        st.error("NESO’s live feed could not be loaded. Try refreshing in a moment.")
        st.stop()
    load_status.update(label="Live demand data ready", state="complete", expanded=False)
model = get_model()
metrics = get_metrics()
latest_date = actual_data["date"].max().normalize()
target_date = latest_date + pd.DateOffset(days=1)

st.title("Grid Demand Lab")
st.caption(f"Forecast for {target_date:%d %b} in 30-minute steps, built from recent settled GB demand and a model trained on 2025 history. Compare with NESO’s forecast when available to see expected peaks and where the two forecasts differ.")
st.caption("Refresh fetches the latest readings and recalculates the forecast. Data is cached for up to 6 hours and is not saved permanently.")
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
chart.index.name = "Settlement period (30 min)"
if neso_count:
    chart["NESO forecast"] = result.set_index("period")["neso_forecast_mw"]
chart_data = chart.rename_axis("period").reset_index().melt(
    "period", var_name="Forecast", value_name="Demand (MW)")
time_labels = settlement_clock_times(target_date.strftime("%Y-%m-%d"), len(result))
tick_positions = [index + 1 for index, label in enumerate(time_labels)
                  if index % 4 == 0 or label.endswith(("BST", "GMT"))]
tick_labels = {position: time_labels[position - 1] for position in tick_positions}
label_expr = " : ".join(f"datum.value === {position} ? '{label}'"
                       for position, label in tick_labels.items()) + " : ''"
forecast_chart = alt.Chart(chart_data).mark_line(point=True).encode(
    x=alt.X("period:Q", title="Time of day (UK)",
            scale=alt.Scale(domain=[1, int(result["period"].max())], clamp=True, nice=False),
            axis=alt.Axis(values=tick_positions, labelExpr=label_expr, labelAngle=0)),
    y=alt.Y("Demand (MW):Q", title="Demand (MW)", scale=alt.Scale(zero=True)),
    color=alt.Color("Forecast:N", title=None),
    tooltip=[alt.Tooltip("period:Q", title="Settlement period", format=".0f"),
             alt.Tooltip("Forecast:N"), alt.Tooltip("Demand (MW):Q", format=",.0f")],
).properties(height=480)
st.altair_chart(forecast_chart, use_container_width=True, theme="streamlit")
fallback_count = int(predicted["used_weekly_fallback"].sum())
if fallback_count:
    st.caption(f"For {fallback_count} periods, the latest actual day is incomplete, so the model used the same half-hour period from 7 days earlier.")
st.metric("Historical holdout MAE · final 28 days of 2025", f"{metrics['model_mae']:,.0f} MW",
          f"{(1 - metrics['model_mae'] / metrics['week_mae']) * 100:.0f}% lower than the previous-week baseline")
st.caption(f"2025 holdout: {metrics['holdout_start']} to {metrics['holdout_end']}. The live model uses 2025 history; its inputs are the latest actual outturns.")
st.link_button("NESO data and licence", "https://www.neso.energy/data-portal/historic-demand-data")
