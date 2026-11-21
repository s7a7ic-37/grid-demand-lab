from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import forecast


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def day_ahead(monkeypatch):
    fetch = Mock(return_value=None)
    monkeypatch.setattr(forecast, "fetch_day_ahead_forecast", fetch)
    return fetch


@pytest.fixture
def source(monkeypatch, day_ahead):
    st.cache_data.clear()
    actual = forecast.load_demand(ROOT / "data" / "demand_2025.csv")
    official = actual.iloc[:0].copy()
    fetch = Mock(return_value=(actual, official))
    monkeypatch.setattr(forecast, "fetch_live_demand", fetch)
    yield fetch
    st.cache_data.clear()


def visit():
    return AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)


def refresh(app):
    return next(button for button in app.button if button.label == "Refresh live data").click().run(timeout=30)


def report_details(app):
    assert not app.exception
    assert len(app.get("arrow_vega_lite_chart")) == 1
    return (
        [metric.value for metric in app.metric],
        [caption.value for caption in app.caption if caption.value.startswith("Feed fetched")],
        len(app.get("download_button")),
    )


def test_failed_refresh_keeps_session_report_and_shared_cache(source):
    app = visit()
    original = report_details(app)
    assert source.call_count == 1

    actual, official = source.return_value
    recovered_actual = actual[actual["date"].lt(actual["date"].max())]
    source.side_effect = [TimeoutError(), (actual.iloc[-48:], official), (recovered_actual, official)]
    for problem in ("could not be loaded", "does not contain enough matching prior periods"):
        calls = source.call_count
        assert report_details(refresh(app)) == original
        assert not app.error
        assert problem in app.warning[0].value
        assert "Showing the last good forecast from this session" in app.warning[0].value
        assert report_details(visit()) == original
        assert source.call_count == calls + 1

    refresh(app)
    assert not app.warning
    assert app.session_state.last_report[1] == pd.Timestamp("2025-12-31")
    assert report_details(visit()) == report_details(app)
    assert source.call_count == 4


def test_first_load_failure_shows_error_without_a_chart(source):
    source.side_effect = TimeoutError()
    app = visit()
    assert not app.exception
    assert not app.get("arrow_vega_lite_chart")
    assert "NESO’s live feed could not be loaded" in app.error[0].value


def test_neso_comparison_is_scoped_to_the_displayed_snapshot(source):
    actual, _ = source.return_value
    official = pd.DataFrame({
        "date": [pd.Timestamp("2025-12-31")], "period": [1], "demand_mw": [27_600],
    })
    source.side_effect = [
        (actual, official), TimeoutError(), (actual[actual["date"].lt(actual["date"].max())], official),
    ]
    app = visit()
    missing = "NESO comparison unavailable for 01 Jan 2026 in this snapshot. Showing the model forecast."
    assert missing in [caption.value for caption in app.caption]
    assert all(not metric.label.startswith("NESO forecast") for metric in app.metric)

    refresh(app)
    assert "Showing the last good forecast from this session" in app.warning[0].value
    assert missing in [caption.value for caption in app.caption]

    refresh(app)
    assert not app.exception
    assert not app.warning
    captions = [caption.value for caption in app.caption]
    assert missing not in captions
    assert (
        "This snapshot has usable NESO values for 1 of 48 half-hours on 31 Dec 2025. "
        "The NESO mean uses those values."
    ) in captions
    assert next(metric for metric in app.metric if metric.label.startswith("NESO forecast")).value == "27,600 MW"


def test_neso_comparison_coverage_counts_only_usable_feed_values(source, day_ahead):
    actual, _ = source.return_value
    other_day = {"SETTLEMENT_DATE": "2025-12-31", "SETTLEMENT_PERIOD": 1, "ND": 27_600}
    records = [{**other_day, "FORECAST_ACTUAL_INDICATOR": "A"}, {**other_day, "FORECAST_ACTUAL_INDICATOR": "F"}]
    records += [
        {"SETTLEMENT_DATE": "2026-01-01", "SETTLEMENT_PERIOD": period, "ND": demand, "FORECAST_ACTUAL_INDICATOR": "F"}
        for period, demand in [(1, 24_000), (2, 0), (3, 26_000)]
    ]
    _, official = forecast.parse_live_records(records)
    source.return_value = actual, official
    app = visit()
    assert not app.exception
    assert (
        "This snapshot has usable NESO values for 2 of 48 half-hours on 01 Jan 2026. "
        "The NESO mean uses those values."
    ) in [caption.value for caption in app.caption]
    assert next(metric for metric in app.metric if metric.label.startswith("NESO forecast")).value == "25,000 MW"
    day_ahead.assert_not_called()


def test_day_ahead_comparison_stays_with_its_snapshot_across_cache_and_failed_refresh(source, day_ahead):
    date = "2026-01-01"
    issue = pd.Timestamp("2025-12-31T16:17:00Z").tz_convert("Europe/London")
    day_ahead.return_value = (
        pd.DataFrame({"date": pd.Timestamp(date), "period": range(1, 49), "demand_mw": 23_000}), issue,
    )
    app = visit()
    original = report_details(app)

    def comparison_caption(report):
        return next(caption.value for caption in report.caption if "via Elexon Insights" in caption.value)

    published = comparison_caption(app)
    assert "31 Dec 2025 at 16:17 GMT" in published
    assert "Contains BMRS data © Elexon Limited copyright and database right 2025." in published
    assert next(metric for metric in app.metric if metric.label == "NESO forecast · day ahead").value == "23,000 MW"
    assert report_details(visit()) == original
    source.side_effect = TimeoutError()
    assert report_details(refresh(app)) == original
    assert comparison_caption(app) == published
    day_ahead.assert_called_once_with(date)


def test_day_ahead_timeout_does_not_prevent_a_model_snapshot(source, day_ahead):
    day_ahead.side_effect = TimeoutError()
    app = visit()
    report_details(app)
    assert not app.error
    assert all(not metric.label.startswith("NESO forecast") for metric in app.metric)
    assert any("Showing the model forecast" in caption.value for caption in app.caption)
    day_ahead.assert_called_once_with("2026-01-01")
