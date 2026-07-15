from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["period", "day_of_week", "month", "is_weekend", "lag_day", "lag_week"]


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
    for period in range(1, 49):
        previous_day = history.get((target - pd.DateOffset(days=1), period))
        previous_week = history.get((target - pd.DateOffset(days=7), period))
        if pd.isna(previous_day) or pd.isna(previous_week):
            continue
        row = pd.DataFrame([{
            "period": period, "day_of_week": target.dayofweek, "month": target.month,
            "is_weekend": int(target.dayofweek >= 5), "lag_day": previous_day,
            "lag_week": previous_week,
        }], columns=FEATURES)
        rows.append((period, float(model.predict(row)[0])))
    if not rows:
        raise ValueError("This date does not have enough prior data to make a forecast.")
    return pd.DataFrame(rows, columns=["period", "forecast_mw"])
