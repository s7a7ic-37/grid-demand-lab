from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import forecast


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def source(monkeypatch):
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
