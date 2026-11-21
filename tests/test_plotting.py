import pandas as pd
import pytest

from forecast import settlement_clock_times
from plotting import demand_chart


def test_chart_has_clock_tooltips_mobile_points_and_fixed_day_domain():
    chart = demand_chart(pd.DataFrame({
        "period": [1, 2], "Forecast": ["Model forecast", "Model forecast"],
        "Demand (MW)": [100, 200],
    }), ["00:00", "00:30"]).to_dict()
    assert chart["layer"][0]["encoding"]["x"]["scale"]["domain"] == [1, 2]
    assert chart["layer"][0]["encoding"]["color"]["legend"]["columns"] == {"expr": "width < 230 ? 1 : 2"}
    assert chart["layer"][1]["mark"]["size"] == {"expr": "width < 600 ? 12 : 58"}
    assert chart["layer"][2]["encoding"]["tooltip"][0]["field"] == "UK time"
    assert chart["params"][0]["select"]["nearest"] is True
    assert "bind" not in chart["params"][0]


def test_chart_keeps_selected_time_and_values_readable_without_a_hover_tooltip():
    chart = demand_chart(pd.DataFrame({
        "period": [1, 2, 1, 2],
        "Forecast": ["Model forecast", "Model forecast", "NESO forecast", "NESO forecast"],
        "Demand (MW)": [1000, 2000, 1500, float("nan")],
    }), ["00:00", "00:30"]).to_dict()
    assert "click" in chart["params"][0]["select"]["on"]
    assert chart["params"][0]["select"]["clear"] is False
    labels = next(rows for rows in chart["datasets"].values() if "label" in rows[0])
    assert labels == [
        {"period": 1, "label": "UK time: 00:00\nModel forecast: 1,000 MW\nNESO forecast: 1,500 MW"},
        {"period": 2, "label": "UK time: 00:30\nModel forecast: 2,000 MW\nNESO forecast: not available"},
    ]


@pytest.mark.parametrize(("date", "periods"), [
    ("2025-03-30", [1, 11, 23, 35]),
    ("2026-01-01", [1, 13, 25, 37]),
    ("2025-10-26", [1, 15, 27, 39]),
])
def test_axis_uses_even_local_clock_anchors_across_day_lengths(date, periods):
    labels = settlement_clock_times(date)
    chart = demand_chart(pd.DataFrame({
        "period": list(range(1, len(labels) + 1)), "Forecast": "Model forecast", "Demand (MW)": 1000,
    }), labels).to_dict()
    axis = chart["layer"][0]["encoding"]["x"]["axis"]
    assert axis["values"] == periods
    assert [labels[period - 1] for period in periods] == ["00:00", "06:00", "12:00", "18:00"]
