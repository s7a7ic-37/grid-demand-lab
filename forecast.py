from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import json
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["period", "day_of_week", "month", "is_weekend", "lag_day", "lag_week"]
LIVE_RESOURCE = "177f6fa4-ae49-4182-81ea-0c6b35f26ca6"


def load_demand(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, usecols=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"])
    data.columns = ["date", "period", "demand_mw"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce").astype("datetime64[ns]")
    data["period"] = pd.to_numeric(data["period"], errors="coerce")
    data["demand_mw"] = pd.to_numeric(data["demand_mw"], errors="coerce")
    data = data.dropna().sort_values(["date", "period"]).reset_index(drop=True)
    data["period"] = data["period"].astype(int)
    if data.empty or not data["period"].between(1, 50).all():
        raise ValueError("The demand file has no usable half-hourly observations.")
    if data.duplicated(["date", "period"]).any():
        raise ValueError("The demand file contains duplicate date and period rows.")
    return data


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
    data["period"] = data["period"].astype(int)
    data = data[data["demand_mw"] > 0].reset_index(drop=True)
    if data.empty and allow_empty:
        return data
    if data.empty or not data["period"].between(1, 50).all():
        raise ValueError("The demand data has no usable half-hourly observations.")
    if data.duplicated(["date", "period"]).any():
        raise ValueError("The demand data contains duplicate date and period rows.")
    return data


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["lag_day"] = result["demand_mw"].shift(48)
    result["lag_week"] = result["demand_mw"].shift(336)
    return result.dropna(subset=FEATURES)


def fit_model(training: pd.DataFrame) -> LGBMRegressor:
    model = LGBMRegressor(n_estimators=250, learning_rate=0.04, num_leaves=20,
                          max_depth=8, min_child_samples=30, verbosity=-1,
                          random_state=21, n_jobs=1)
    model.fit(training[FEATURES], training["demand_mw"])
    return model


def evaluate(data: pd.DataFrame, holdout_days: int = 28) -> tuple[dict[str, float | str], LGBMRegressor]:
    featured = make_features(data)
    cutoff = featured["date"].max().normalize() - pd.Timedelta(days=holdout_days - 1)
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
    }
    return metrics, model


def forecast_day(data: pd.DataFrame, model: LGBMRegressor, date: str) -> pd.DataFrame:
    target = pd.Timestamp(date).normalize()
    history = data.set_index(["date", "period"])["demand_mw"]
    rows = []
    start = target.tz_localize("Europe/London")
    end = (target + pd.DateOffset(days=1)).tz_localize("Europe/London")
    period_count = int((end - start).total_seconds() / 1800)
    for period in range(1, period_count + 1):
        source_period = min(period, 48)
        previous_day = history.get((target - pd.DateOffset(days=1), source_period))
        previous_week = history.get((target - pd.DateOffset(days=7), source_period))
        if pd.isna(previous_week):
            continue
        used_weekly_fallback = pd.isna(previous_day)
        if used_weekly_fallback:
            previous_day = previous_week
        row = pd.DataFrame([{
            "period": period, "day_of_week": target.dayofweek, "month": target.month,
            "is_weekend": int(target.dayofweek >= 5), "lag_day": previous_day,
            "lag_week": previous_week,
        }], columns=FEATURES)
        rows.append((period, float(model.predict(row)[0]), used_weekly_fallback))
    if not rows:
        raise ValueError("This date does not have enough prior data to make a forecast.")
    return pd.DataFrame(rows, columns=["period", "forecast_mw", "used_weekly_fallback"])


def settlement_clock_times(date: str, period_count: int) -> list[str]:
    starts = pd.date_range(pd.Timestamp(date).normalize().tz_localize("Europe/London"),
                           periods=period_count, freq="30min")
    labels = [start.strftime("%H:%M") for start in starts]
    repeated = pd.Series(labels).duplicated(keep=False).tolist()
    return [f"{label} {start.tzname()}" if duplicate else label
            for label, start, duplicate in zip(labels, starts, repeated)]
