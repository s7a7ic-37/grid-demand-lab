import altair as alt
import pandas as pd


def demand_chart(data: pd.DataFrame, time_labels: list[str]) -> alt.LayerChart:
    data = data.copy()
    data["UK time"] = data["period"].map(dict(enumerate(time_labels, start=1)))
    ticks = [index + 1 for index, label in enumerate(time_labels)
             if index % 4 == 0 or label.endswith(("BST", "GMT"))]
    labels = " : ".join(f"datum.value === {period} ? '{time_labels[period - 1]}'"
                        for period in ticks) + " : ''"
    base = alt.Chart(data).encode(
        x=alt.X("period:Q", title="Time of day (UK)",
                scale=alt.Scale(domain=[1, len(time_labels)], clamp=True, nice=False),
                axis=alt.Axis(values=ticks, labelExpr=labels, labelAngle=0)),
        y=alt.Y("Demand (MW):Q", title="Demand (MW)", scale=alt.Scale(zero=True)),
        color=alt.Color("Forecast:N", title=None, legend=alt.Legend(orient="bottom")),
    )
    points = base.mark_circle(size=alt.ExprRef(expr="width < 600 ? 18 : 58"),
                              stroke="#d6ecff", strokeWidth=.65)
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["period"], empty=False)
    hover = alt.Chart(data).transform_pivot(
        "Forecast", value="Demand (MW)", groupby=["period", "UK time"]
    ).mark_rule(color="#a6c3b8").encode(
        x="period:Q",
        opacity=alt.condition(nearest, alt.value(.35), alt.value(0)),
        tooltip=[alt.Tooltip("UK time:N")]
                + [alt.Tooltip(series, type="quantitative", format=",.0f", title=f"{series} (MW)")
                   for series in data["Forecast"].unique()],
    ).add_params(nearest)
    return alt.layer(base.mark_line(), points, hover).properties(height=480)
