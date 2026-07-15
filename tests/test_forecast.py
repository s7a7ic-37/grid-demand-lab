import pandas as pd
import pytest

from forecast import forecast_day, load_demand, make_features


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


def test_forecast_requires_seven_days_of_prior_history():
    history = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"], format="%Y-%m-%d"),
                            "period": [1], "demand_mw": [100]})

    class Model:
        def predict(self, values):
            return [100]

    with pytest.raises(ValueError, match="enough prior data"):
        forecast_day(history, Model(), "2025-01-02")
