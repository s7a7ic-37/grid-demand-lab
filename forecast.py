from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import numpy as np
import json
from lightgbm import Booster, LGBMRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["period", "day_of_week", "month", "is_weekend", "lag_day", "lag_week"]
LIVE_RESOURCE = "177f6fa4-ae49-4182-81ea-0c6b35f26ca6"


def load_demand(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, usecols=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"])
    data.columns = ["date", "period", "demand_mw"]
    return clean_demand(data)


def fetch_live_demand() -> tuple[pd.DataFrame, pd.DataFrame]:
    query = urlencode({"resource_id": LIVE_RESOURCE, "limit": 5000})
    with urlopen(f"https://api.neso.energy/api/3/action/datastore_search?{query}", timeout=20) as response:
        payload = json.loads(response.read())
    result = payload.get("result", {})
    if not payload.get("success") or not isinstance(result.get("records"), list):
        raise ValueError("NESO returned an invalid demand-data response.")
    if result.get("total", 0) > len(result["records"]):
        raise ValueError("NESO returned an incomplete live demand-data page.")
    return parse_live_records(result["records"])


def parse_live_records(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.DataFrame(records)
    if records.empty:
        raise ValueError("NESO returned no live demand observations.")
    required = {"SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND", "FORECAST_ACTUAL_INDICATOR"}
    if not required <= set(records.columns):
        raise ValueError("NESO demand data is missing required columns.")
    records = records.rename(columns={
        "SETTLEMENT_DATE": "date", "SETTLEMENT_PERIOD": "period", "ND": "demand_mw",
    })
    actual = records[records["FORECAST_ACTUAL_INDICATOR"].eq("A")]
    forecast = records[records["FORECAST_ACTUAL_INDICATOR"].eq("F")]
    return clean_demand(actual), clean_demand(forecast, allow_empty=True)


def clean_demand(data: pd.DataFrame, allow_empty: bool = False) -> pd.DataFrame:
    data = data[["date", "period", "demand_mw"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").astype("datetime64[ns]")
    data["period"] = pd.to_numeric(data["period"], errors="coerce")
    data["demand_mw"] = pd.to_numeric(data["demand_mw"], errors="coerce")
    data = data.dropna().sort_values(["date", "period"]).reset_index(drop=True)
    if not np.isfinite(data[["period", "demand_mw"]]).all().all():
        raise ValueError("Demand values and periods must be finite numbers.")
    if data["period"].mod(1).ne(0).any():
        raise ValueError("Settlement periods must be whole numbers.")
    data["period"] = data["period"].astype(int)
    data = data[data["demand_mw"] > 0].reset_index(drop=True)
    if data.empty and allow_empty:
        return data
    if data.empty or not data["period"].between(1, 50).all():
        raise ValueError("The demand data has no usable half-hourly observations.")
    day_lengths = {day: len(settlement_clock_times(day.strftime("%Y-%m-%d")))
                   for day in data["date"].drop_duplicates()}
    if data["period"].gt(data["date"].map(day_lengths)).any():
        raise ValueError("Settlement periods must fit the UK date's 46, 48 or 50 half-hour periods.")
    if data.duplicated(["date", "period"]).any():
        raise ValueError("The demand data contains duplicate date and period rows.")
    return data


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    history = data.set_index(["date", "period"])["demand_mw"]
    for name, days in (("lag_day", 1), ("lag_week", 7)):
        lag_dates = result["date"].to_numpy(dtype="datetime64[ns]") - np.timedelta64(days, "D")
        keys = pd.MultiIndex.from_arrays([lag_dates, result["period"]])
        result[name] = history.reindex(keys).to_numpy()
    return result.dropna(subset=FEATURES)


def fit_model(training: pd.DataFrame) -> LGBMRegressor:
    model = LGBMRegressor(n_estimators=250, learning_rate=0.04, num_leaves=20,
                          max_depth=8, min_child_samples=30, verbosity=-1,
                          random_state=21, n_jobs=1)
    model.fit(training[FEATURES], training["demand_mw"])
    return model


def evaluate(data: pd.DataFrame, holdout_days: int = 28) -> tuple[dict[str, float | str], LGBMRegressor]:
    featured = make_features(data)
    cutoff = featured["date"].max().normalize() - pd.DateOffset(days=holdout_days - 1)
    train, test = featured[featured["date"] < cutoff], featured[featured["date"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Not enough history for a time-based holdout.")
    model = fit_model(train)
    metrics = {
        "model_mae": mean_absolute_error(test["demand_mw"], model.predict(test[FEATURES])),
        "day_mae": mean_absolute_error(test["demand_mw"], test["lag_day"]),
        "week_mae": mean_absolute_error(test["demand_mw"], test["lag_week"]),
        "holdout_start": cutoff.strftime("%Y-%m-%d"),
        "holdout_end": test["date"].max().strftime("%Y-%m-%d"),
        "training_end": train["date"].max().strftime("%Y-%m-%d"),
        "holdout_periods": len(test),
    }
    return metrics, model


def quarterly_backtests(data: pd.DataFrame) -> list[dict]:
    reports = []
    for end in pd.date_range(data["date"].min(), data["date"].max(), freq=pd.offsets.QuarterEnd()):
        scores, _ = evaluate(data[data["date"] <= end])
        reports.append(scores)
    if not reports:
        raise ValueError("At least one complete quarter is needed for quarterly backtests.")
    return reports


def forecast_day(data: pd.DataFrame, model: Booster | LGBMRegressor, date: str) -> pd.DataFrame:
    target = pd.Timestamp(date).normalize()
    history = data.set_index(["date", "period"])["demand_mw"]
    rows = []
    period_count = len(settlement_clock_times(date))
    for period in range(1, period_count + 1):
        # shortcut: extra clock-change periods reuse period 48; use clock-aligned lags for operational forecasting.
        source_period = min(period, 48)
        previous_day = history.get((target - pd.DateOffset(days=1), source_period))
        previous_week = history.get((target - pd.DateOffset(days=7), source_period))
        if pd.isna(previous_week):
            continue
        used_weekly_fallback = pd.isna(previous_day)
        if used_weekly_fallback:
            previous_day = previous_week
        rows.append({
            "period": period, "day_of_week": target.dayofweek, "month": target.month,
            "is_weekend": int(target.dayofweek >= 5), "lag_day": previous_day,
            "lag_week": previous_week, "used_weekly_fallback": used_weekly_fallback,
        })
    if not rows:
        raise ValueError("This date does not have enough prior data to make a forecast.")
    result = pd.DataFrame(rows)
    result["forecast_mw"] = model.predict(result[FEATURES])
    return result[["period", "forecast_mw", "used_weekly_fallback"]]


def settlement_clock_times(date: str) -> list[str]:
    start = pd.Timestamp(date).normalize().tz_localize("Europe/London")
    starts = pd.date_range(start, start + pd.DateOffset(days=1), freq="30min", inclusive="left")
    labels = [start.strftime("%H:%M") for start in starts]
    repeated = pd.Series(labels).duplicated(keep=False).tolist()
    return [f"{label} {start.tzname()}" if duplicate else label
            for label, start, duplicate in zip(labels, starts, repeated)]


def forecast_csv(result: pd.DataFrame, date: str, fetched_at: pd.Timestamp) -> str:
    snapshot = result[["period", "forecast_mw", "neso_forecast_mw", "used_weekly_fallback"]].copy()
    snapshot.insert(0, "date", date)
    clocks = dict(enumerate(settlement_clock_times(date), start=1))
    snapshot.insert(2, "uk_time", snapshot["period"].map(clocks))
    snapshot["fetched_at"] = fetched_at.isoformat()
    return snapshot.to_csv(index=False)
