import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_TASKS = 1000

TASK_TYPES = ["design", "development", "testing", "review", "deployment", "research"]
COMPLEXITY = ["low", "medium", "high"]

COMPLEXITY_MULTIPLIER = {"low": 0.85, "medium": 1.0, "high": 1.35}
TASK_TYPE_MULTIPLIER = {
    "design": 1.1,
    "development": 1.25,
    "testing": 0.95,
    "review": 0.80,
    "deployment": 1.15,
    "research": 1.40,
}


def generate_data(n: int = N_TASKS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    task_type = rng.choice(TASK_TYPES, size=n)
    complexity = rng.choice(COMPLEXITY, size=n)
    team_size = rng.integers(1, 8, size=n)
    dependency_count = rng.integers(0, 6, size=n)
    is_external = rng.integers(0, 2, size=n)

    optimistic = rng.uniform(1, 5, size=n)
    most_likely = optimistic + rng.uniform(1, 7, size=n)
    pessimistic = most_likely + rng.uniform(1, 10, size=n)

    expected = (optimistic + 4 * most_likely + pessimistic) / 6
    variance = ((pessimistic - optimistic) / 6) ** 2

    type_mult = np.array([TASK_TYPE_MULTIPLIER[t] for t in task_type])
    comp_mult = np.array([COMPLEXITY_MULTIPLIER[c] for c in complexity])
    dep_penalty = 1 + (dependency_count * 0.05)
    external_penalty = 1 + (is_external * 0.15)
    noise = rng.normal(1.0, 0.1, size=n)

    actual_duration = (
        expected * type_mult * comp_mult * dep_penalty * external_penalty * noise
    )
    actual_duration = np.clip(actual_duration, optimistic, pessimistic * 1.5)

    df = pd.DataFrame({
        "task_type": task_type,
        "complexity": complexity,
        "team_size": team_size,
        "dependency_count": dependency_count,
        "is_external": is_external,
        "optimistic": optimistic.round(2),
        "most_likely": most_likely.round(2),
        "pessimistic": pessimistic.round(2),
        "pert_expected": expected.round(2),
        "pert_variance": variance.round(4),
        "actual_duration": actual_duration.round(2),
    })

    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent / "sample_tasks.csv"
    df = generate_data()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} tasks → {output_path}")
    print(df.describe().round(2))