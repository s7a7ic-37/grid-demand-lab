import pandas as pd

from plotting import demand_chart


def test_chart_has_clock_tooltips_mobile_points_and_fixed_day_domain():
    chart = demand_chart(pd.DataFrame({
        "period": [1, 2], "Forecast": ["Model forecast", "Model forecast"],
        "Demand (MW)": [100, 200],
    }), ["00:00", "00:30"]).to_dict()
    assert chart["layer"][0]["encoding"]["x"]["scale"]["domain"] == [1, 2]
    assert chart["layer"][1]["mark"]["size"] == {"expr": "width < 600 ? 18 : 58"}
    assert chart["layer"][2]["encoding"]["tooltip"][0]["field"] == "UK time"
    assert chart["params"][0]["select"]["nearest"] is True
    assert "bind" not in chart["params"][0]
