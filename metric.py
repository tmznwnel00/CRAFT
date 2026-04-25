import pprint
import os 
import statistics

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

from be_great.metrics import (
    BasicStatistics,
    ColumnPairTrends,
    ColumnShapes,
    DiscriminatorMetric,
    DistanceToClosestRecord,
    MLEfficiency,
)

# adult
# COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
# NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss","hours-per-week"]

# asia
COLS = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"]
NUM_COLS = []

# healthcare
# COLS = ["A", "C", "D", "H", "I", "O", "T"]
# NUM_COLS = ["D", "I", "O", "T"]

CAT_COLS = [c for c in COLS if c not in NUM_COLS]


def build_real_data() -> pd.DataFrame:
    # from ucimlrepo import fetch_ucirepo
    # adult = fetch_ucirepo(id=2)
    # real_data = adult.data.features.copy()
    # real_data["income"] = adult.data.targets["income"]
    import bnlearn as bn
    real_data = bn.import_example("asia")
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # real_data = pd.read_csv(os.path.join(base_dir, "examples/healthcare.txt"), sep=" ")

    real_data = real_data[COLS].copy()
    # real_data = real_data.iloc[32561:].copy() # adult
    # real_data["income"] = real_data["income"].astype(str).str.replace(".", "", regex=False)
    real_data = real_data.iloc[8000:].copy() # asia
    # real_data = real_data.iloc[1600:].copy() # healthcare
    
    
    # Ensure stable dtypes for metric encoders
    for col in NUM_COLS:
        real_data[col] = pd.to_numeric(real_data[col], errors="coerce")
    for col in CAT_COLS:
        real_data[col] = real_data[col].fillna("missing").astype(str)
    return real_data.reset_index(drop=True)


def build_synthetic_data(csv_path: str) -> pd.DataFrame:
    synthetic_data = pd.read_csv(csv_path, index_col=0)
    synthetic_data = synthetic_data[COLS].copy()
    # synthetic_data["income"] = synthetic_data["income"].astype(str).str.replace(".", "", regex=False)
    # Match dtype handling with real_data to avoid mixed int/str categories
    for col in NUM_COLS:
        synthetic_data[col] = pd.to_numeric(synthetic_data[col], errors="coerce")
    for col in CAT_COLS:
        synthetic_data[col] = synthetic_data[col].fillna("missing").astype(str)
    return synthetic_data.reset_index(drop=True)


def run_metrics(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    output_path: str = "metric_results.txt",
    label_col: str = "income",
) -> dict:
    def is_continuous_target(series: pd.Series) -> bool:
        numeric_series = pd.to_numeric(series, errors="coerce")
        valid = numeric_series.dropna()
        if valid.empty:
            return False
        # 연속형 여부를 실용적으로 판별하기 위한 휴리스틱:
        # 숫자형이면서 고유값이 충분히 많으면 회귀로 처리
        return valid.nunique() > 10

    is_regression = is_continuous_target(real_data[label_col])
    model_cls = RandomForestRegressor if is_regression else RandomForestClassifier
    score_fn = r2_score if is_regression else accuracy_score

    metrics = {
        # "ColumnShapes": ColumnShapes().compute(real_data, synthetic_data),
        # "ColumnPairTrends": ColumnPairTrends().compute(real_data, synthetic_data),
        # "BasicStatistics": BasicStatistics().compute(real_data, synthetic_data),
        # "DiscriminatorMetric": DiscriminatorMetric().compute(real_data, synthetic_data),
        "MLEfficiency": MLEfficiency(
            model=model_cls,
            metric=score_fn,
            model_params={"n_estimators": 100},
        ).compute(real_data, synthetic_data, label_col=label_col),
        # "DistanceToClosestRecord": DistanceToClosestRecord().compute(real_data, synthetic_data),
    }

    print("\n=== Metric Results ===")
    lines = ["=== Metric Results ===", ""]
    for name, result in metrics.items():
        print(f"\n[{name}]")
        pprint.pprint(result, sort_dicts=False)
        lines.append(f"[{name}]")
        lines.append(pprint.pformat(result, sort_dicts=False))
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"\nSaved metric results to: {output_path}")
    return metrics


def save_combined_results(all_metrics: list[dict], output_path: str) -> None:
    lines = ["=== Metric Results ===", ""]

    for run_idx, metrics in enumerate(all_metrics, start=1):
        lines.append(f"[Run {run_idx}]")
        for name, result in metrics.items():
            lines.append(f"[{name}]")
            lines.append(pprint.pformat(result, sort_dicts=False))
            lines.append("")

    # 실험 반복(run) 기준으로 각 metric의 평균/표준편차를 집계
    if all_metrics:
        lines.append("[Aggregate]")
        for metric_name in all_metrics[0].keys():
            per_run_scores = []
            for metrics in all_metrics:
                metric_result = metrics[metric_name]
                if isinstance(metric_result, dict) and "mle_mean" in metric_result:
                    per_run_scores.append(float(metric_result["mle_mean"]))
            if per_run_scores:
                aggregate_result = {
                    "per_run_scores": per_run_scores,
                    "mean": statistics.mean(per_run_scores),
                    "std": statistics.pstdev(per_run_scores) if len(per_run_scores) > 1 else 0.0,
                }
                lines.append(f"[{metric_name}]")
                lines.append(pprint.pformat(aggregate_result, sort_dicts=False))
                lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"\nSaved combined metric results to: {output_path}")


if __name__ == "__main__":
    real_data = build_real_data()
    os.makedirs("result/asia/topology/", exist_ok=True)
    all_metrics = []
    for i in range(1, 6):
        synthetic_data = build_synthetic_data(f"result/asia/topology/asia_test_topology{i}.csv")
        metrics = run_metrics(
            real_data,
            synthetic_data,
            output_path=f"result/asia/topology/metric_asia_topology{i}.txt",
            label_col="dysp",
        )
        all_metrics.append(metrics)

    save_combined_results(all_metrics, output_path="result/asia/topology/combined_metric.txt")