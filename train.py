import json
from pathlib import Path

from forecast import evaluate, fit_model, load_demand, make_features, quarterly_backtests

ROOT = Path(__file__).parent


def main() -> None:
    data = load_demand(ROOT / "data" / "demand_2025.csv")
    metrics, _ = evaluate(data)
    report = {**metrics, "backtests": quarterly_backtests(data)}
    current_model = fit_model(make_features(data))
    (ROOT / "models").mkdir(exist_ok=True)
    current_model.booster_.save_model(str(ROOT / "models" / "demand_model.txt"))
    (ROOT / "models" / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
