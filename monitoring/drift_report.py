from pathlib import Path
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

TRAINING_DATA_PATH = Path(__file__).parent.parent / "data" / "sample_tasks.csv"
REPORT_OUTPUT_PATH = Path(__file__).parent / "drift_report.html"

FEATURE_COLS = [
    "team_size",
    "dependency_count",
    "is_external",
    "optimistic",
    "most_likely",
    "pessimistic",
    "pert_expected",
]


def simulate_production_data(reference: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    production = reference.sample(n, replace=True, random_state=99).copy()

    production["team_size"] = (
        production["team_size"] * rng.uniform(1.2, 1.8, size=n)
    ).clip(1, 20).astype(int)
    
    production["pessimistic"] = (
        production["pessimistic"] * rng.uniform(1.1, 1.4, size=n)
    ).round(2)
    
    production["pert_expected"] = (
        (production["optimistic"] + 4 * production["most_likely"] + production["pessimistic"]) / 6
    ).round(2)

    return production


def run_drift_report(save_path: Path = REPORT_OUTPUT_PATH):
    print("Loading training (reference) data...")
    reference = pd.read_csv(TRAINING_DATA_PATH)[FEATURE_COLS]

    print("Simulating production (current) data with drift...")
    current = simulate_production_data(reference)

    print("Running Evidently drift analysis...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference, current_data=current[FEATURE_COLS])
    report.save_html(str(save_path))
    print(f"Drift report saved → {save_path}")

    result = report.as_dict()
    drift_metric = result["metrics"][0]["result"]
    n_drifted = drift_metric.get("number_of_drifted_columns", 0)
    n_total = drift_metric.get("number_of_columns", len(FEATURE_COLS))
    share = drift_metric.get("share_of_drifted_columns", 0)

    print(f"\nDrift Summary:")
    print(f"  Drifted features: {n_drifted}/{n_total} ({share*100:.1f}%)")
    print(f"  Dataset drift detected: {drift_metric.get('dataset_drift', False)}")

    if drift_metric.get("dataset_drift"):
        print("\n  Significant drift detected. Model retraining recommended.")
    else:
        print("\n  Drift within acceptable range. Model still valid.")


if __name__ == "__main__":
    run_drift_report()