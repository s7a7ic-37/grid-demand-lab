import pandas as pd
import pytest

from forecast import forecast_day, load_demand, make_features, parse_live_records


def test_features_use_only_previous_day_and_previous_week():
    rows = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=8).astype("datetime64[ns]").repeat(48),
                         "period": list(range(1, 49)) * 8,
                         "demand_mw": range(384)})
    featured = make_features(rows)
    assert featured.iloc[0]["lag_day"] == 288
    assert featured.iloc[0]["lag_week"] == 0
    assert featured.iloc[0]["demand_mw"] == 336


def test_invalid_and_duplicate_rows_are_rejected(tmp_path):
    path = tmp_path / "demand.csv"
    path.write_text('SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND\n2025-01-01,1,100\n2025-01-01,1,101\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_demand(path)


def test_live_records_keep_outturns_and_operator_forecasts_separate():
    actual, forecast = parse_live_records([
        {"SETTLEMENT_DATE": "2026-10-30", "SETTLEMENT_PERIOD": 1, "ND": 24000,
         "FORECAST_ACTUAL_INDICATOR": "A"},
        {"SETTLEMENT_DATE": "2026-11-01", "SETTLEMENT_PERIOD": 1, "ND": 25000,
         "FORECAST_ACTUAL_INDICATOR": "F"},
    ])
    assert actual.iloc[0]["demand_mw"] == 24000
    assert forecast.iloc[0]["demand_mw"] == 25000


def test_live_records_keep_actual_data_when_feed_forecast_is_empty():
    actual, forecast = parse_live_records([
        {"SETTLEMENT_DATE": "2026-10-30", "SETTLEMENT_PERIOD": 1, "ND": 24000,
         "FORECAST_ACTUAL_INDICATOR": "A"},
        {"SETTLEMENT_DATE": "2026-11-01", "SETTLEMENT_PERIOD": 1, "ND": 0,
         "FORECAST_ACTUAL_INDICATOR": "F"},
    ])
    assert len(actual) == 1
    assert forecast.empty


def test_forecast_requires_seven_days_of_prior_history():
    history = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"], format="%Y-%m-%d"),
                            "period": [1], "demand_mw": [100]})

    class Model:
        def predict(self, values):
            return [100]

    with pytest.raises(ValueError, match="enough prior data"):
        forecast_day(history, Model(), "2025-01-02")


def test_live_forecast_fills_incomplete_latest_day_from_last_week():
    history = pd.DataFrame({
        "date": pd.date_range("2026-10-23", periods=9).repeat(48).astype("datetime64[ns]"),
        "period": list(range(1, 49)) * 9,
        "demand_mw": range(432),
    })
    history = history[~((history["date"] == "2026-10-31") & (history["period"] > 15))]

    class Model:
        def predict(self, values):
            return [20000]

    forecast = forecast_day(history, Model(), "2026-11-01")
    assert len(forecast) == 48
    assert forecast["used_weekly_fallback"].sum() == 33
