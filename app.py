from pathlib import Path
import json
from uuid import uuid4

import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Grid Demand Lab", page_icon="⚡", layout="wide")


with st.container(horizontal=True):
    refresh_requested = st.button("Refresh live data")
    st.link_button("← All projects", "https://portfolio.artemislab.co.uk/#projects")

report_key = f"grid-report-{uuid4().hex}"
st.markdown("""
<style>
.grid-loader { display:grid; min-height:52vh; place-content:center; justify-items:center; gap:16px; padding:32px 20px;
  border:1px solid #25443f; border-radius:14px; background:radial-gradient(ellipse at center,#102321,#0b1114 72%);
  color:#e2f4ed; text-align:center; }
.grid-loader__ring { width:44px; height:44px; border:3px solid #25443f; border-top-color:#55e0bd;
  border-radius:50%; box-shadow:0 0 20px #55e0bd55; animation:grid-spin .9s linear infinite; }
.grid-loader__title { margin:0; font-size:18px; font-weight:650; }
.grid-loader__detail { max-width:34rem; margin:0; color:#a6c3b8; font-size:14px; line-height:1.6; }
.grid-loader__track { width:min(260px,65vw); height:4px; overflow:hidden; border-radius:4px; background:#25443f; }
.grid-loader__track::after { display:block; width:42%; height:100%; border-radius:4px; background:#55e0bd;
  content:""; animation:grid-sweep 1.3s ease-in-out infinite alternate; }
@keyframes grid-spin { to { transform:rotate(360deg); } }
@keyframes grid-sweep { to { transform:translateX(125%); } }
:where(body:has(.st-key-grid-report [data-testid="stVegaLiteChart"] g.mark-symbol.role-mark path)) .st-key-grid-report > [data-testid="stElementContainer"] { animation:grid-enter .52s cubic-bezier(.2,.7,.2,1) both; animation-delay:.8s; }
.st-key-grid-report > [data-testid="stElementContainer"]:first-child { animation-delay:0s; }
.st-key-grid-report > [data-testid="stElementContainer"]:nth-child(2) { animation-delay:.08s; }
.st-key-grid-report > [data-testid="stElementContainer"]:nth-child(3) { animation-delay:.16s; }
.st-key-grid-report > [data-testid="stElementContainer"]:nth-child(4) { animation-delay:.24s; }
.st-key-grid-report > [data-testid="stElementContainer"]:nth-child(5) { animation-delay:.32s; }
.st-key-grid-report > [data-testid="stElementContainer"]:nth-child(6) { animation-delay:.40s; }
.st-key-grid-report > [data-testid="stElementContainer"]:nth-child(7) { animation-delay:.48s; }
.st-key-grid-report > [data-testid="stElementContainer"]:nth-child(8) { animation-delay:.56s; }
.st-key-grid-report > [data-testid="stElementContainer"]:nth-child(9) { animation-delay:.64s; }
.st-key-grid-report > [data-testid="stElementContainer"]:nth-child(10) { animation-delay:.72s; }
@keyframes grid-enter { from { opacity:0; transform:translateY(9px); } to { opacity:1; transform:translateY(0); } }
@media(prefers-reduced-motion:reduce) { .grid-loader__ring,.grid-loader__track::after { animation:none; } }
@media(max-width:640px) {
  .grid-loader { min-height:58vh; padding:24px 16px; }
  .st-key-grid-report > [data-testid="stElementContainer"]:nth-child(2) { animation-delay:.06s; }
  .st-key-grid-report > [data-testid="stElementContainer"]:nth-child(3) { animation-delay:.12s; }
  .st-key-grid-report > [data-testid="stElementContainer"]:nth-child(4) { animation-delay:.18s; }
  .st-key-grid-report > [data-testid="stElementContainer"]:nth-child(5) { animation-delay:.24s; }
  .st-key-grid-report > [data-testid="stElementContainer"]:nth-child(6) { animation-delay:.30s; }
  .st-key-grid-report > [data-testid="stElementContainer"]:nth-child(7) { animation-delay:.36s; }
  .st-key-grid-report > [data-testid="stElementContainer"]:nth-child(8) { animation-delay:.42s; }
  .st-key-grid-report > [data-testid="stElementContainer"]:nth-child(9) { animation-delay:.48s; }
  .st-key-grid-report > [data-testid="stElementContainer"]:nth-child(10) { animation-delay:.54s; }

}
@media(prefers-reduced-motion:reduce) { .st-key-grid-report > [data-testid="stElementContainer"] { animation:none; } }
/* Keep layout measurable and reveal the report once the chart has data marks. */
[class*="st-key-grid-report-"] { visibility:hidden; }
[class*="st-key-grid-report-"] [data-testid="stMetricLabel"] { visibility:inherit; }
body:has(.st-key-grid-report [data-testid="stVegaLiteChart"] g.mark-symbol.role-mark path) .st-key-grid-report { visibility:visible; }
body:has(.st-key-grid-report [data-testid="stVegaLiteChart"] g.mark-symbol.role-mark path) [data-testid="stElementContainer"]:has(.grid-loader) { display:none; }
</style>
""".replace(".st-key-grid-report", f".st-key-{report_key}"), unsafe_allow_html=True)

loader = st.empty()
loader.markdown("""
<div class="grid-loader" role="status" aria-live="polite" aria-busy="true">
  <span class="grid-loader__ring" aria-hidden="true"></span>
  <p class="grid-loader__title">Loading live grid demand</p>
  <p class="grid-loader__detail">Fetching the latest NESO readings and preparing one day of half-hourly forecasts. After inactivity, the service may take up to a minute to start.</p>
  <div class="grid-loader__track" aria-hidden="true"></div>
</div>
""", unsafe_allow_html=True)

# Send the loader before importing the model and dataframe libraries on a cold worker.
import pandas as pd
from lightgbm import Booster

from forecast import fetch_live_demand, forecast_day, settlement_clock_times
from plotting import demand_chart


@st.cache_data(ttl=21600, show_spinner=False)
def get_live_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    actual, forecast = fetch_live_demand()
    return actual, forecast, pd.Timestamp.now(tz="Europe/London")


@st.cache_resource(show_spinner=False)
def get_model() -> Booster:
    return Booster(model_file=str(ROOT / "models" / "demand_model.txt"))


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    return json.loads((ROOT / "models" / "metrics.json").read_text())


if refresh_requested:
    get_live_data.clear()

try:
    actual_data, neso_forecast, fetched_at = get_live_data()
except Exception:
    loader.error("NESO’s live feed could not be loaded. Try refreshing in a moment.")
    st.stop()
model = get_model()
metrics = get_metrics()
latest_date = actual_data["date"].max().normalize()
target_date = latest_date + pd.DateOffset(days=1)

try:
    predicted = forecast_day(actual_data, model, target_date.strftime("%Y-%m-%d"))
except ValueError:
    loader.error("The feed does not contain enough matching prior periods to prepare a forecast. Please try again when more settled readings are available.")
    st.stop()
time_labels = settlement_clock_times(target_date.strftime("%Y-%m-%d"))
period_count = len(time_labels)
official = neso_forecast[neso_forecast["date"].eq(target_date)][["period", "demand_mw"]]
periods = pd.DataFrame({"period": range(1, period_count + 1)})
result = periods.merge(predicted, on="period", how="left").merge(official, on="period", how="left")
result = result.rename(columns={"demand_mw": "neso_forecast_mw"})

neso_count = int(result["neso_forecast_mw"].notna().sum())
chart = result.set_index("period")[["forecast_mw"]].rename(columns={"forecast_mw": "Model forecast"})
chart.index.name = "Settlement period (30 min)"
if neso_count:
    chart["NESO forecast"] = result.set_index("period")["neso_forecast_mw"]
chart_data = chart.rename_axis("period").reset_index().melt(
    "period", var_name="Forecast", value_name="Demand (MW)")
forecast_chart = demand_chart(chart_data, time_labels)
fallback_count = int(predicted["used_weekly_fallback"].sum())

with st.container(key=report_key):
    st.title("Grid Demand Lab")
    st.caption(f"One-day demand estimate for {target_date:%d %b %Y}: {period_count} half-hour periods in UK local time. LightGBM estimates the day after the latest settled GB demand readings. Compare expected peaks with NESO’s own forecast when available.")
    if len(predicted) < period_count:
        st.caption(f"Model coverage: {len(predicted)} of {period_count} periods. Gaps have insufficient prior history; the mean uses the available predictions.")
    st.altair_chart(forecast_chart, use_container_width=True, theme="streamlit")
    st.caption(f"Feed fetched {fetched_at:%d %b %Y at %H:%M %Z}. Refresh fetches new readings and recalculates; the current response is cached for up to 6 hours and is not saved permanently.")
    st.caption(f"Latest settled demand in this feed: {latest_date:%d %b %Y}. A delayed feed means the plotted day may also be behind today.")
    st.metric(f"Mean model estimate · {target_date:%d %b}", f"{result['forecast_mw'].mean():,.0f} MW")
    if neso_count:
        st.metric("NESO forecast · same day", f"{result['neso_forecast_mw'].mean():,.0f} MW")
        if neso_count < period_count:
            st.caption(f"NESO has published {neso_count} of {period_count} half-hour forecasts for this day; its mean uses those available periods.")
    else:
        st.caption("NESO’s forecast values are currently missing; showing the model forecast on its own.")
    if fallback_count:
        st.caption(f"For {fallback_count} periods, the latest actual day is incomplete, so the model used the same half-hour period from 7 days earlier.")
    st.metric("Historical holdout MAE · final 28 days of 2025", f"{metrics['model_mae']:,.0f} MW",
              f"{(1 - metrics['model_mae'] / metrics['week_mae']) * 100:.0f}% lower than the previous-week baseline")
    st.caption(f"2025 holdout: {metrics['holdout_start']} to {metrics['holdout_end']}. The live model uses 2025 history; its inputs are the latest actual outturns.")
    st.link_button("NESO data and licence", "https://www.neso.energy/data-portal/historic-demand-data")
    st.link_button("Source ↗", "https://github.com/s7a7ic-37/grid-demand-lab")
