import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


INPUT_CSV = os.path.join("output", "my_log_analysis.csv")


def build_proxy_label(df: pd.DataFrame) -> pd.Series:
    """Independent weak label used for fair comparison across methods."""
    severe_level = df["level"].isin(["ERROR", "CRITICAL", "FATAL"])
    failure_text = df["message"].astype(str).str.contains(
        r"exception|failed|timeout|error|unavailable|route to host",
        case=False,
        regex=True,
        na=False,
    )
    return (severe_level | failure_text).astype(int)


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> int:
    if not os.path.exists(INPUT_CSV):
        print(f"Missing file: {INPUT_CSV}")
        print("Run aiops_log_analysis.py first.")
        return 1

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["if_flag", "zscore_flag", "level", "message"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing required columns in {INPUT_CSV}: {missing}")
        return 1

    y_ref = build_proxy_label(df)
    final_combined = (
        (df["anomaly_score"] == -1).astype(int)
        if "anomaly_score" in df.columns
        else df["status"].astype(str).str.contains("Anomaly", na=False).astype(int)
    )

    methods = {
        "IsolationForest only (if_flag)": df["if_flag"].fillna(0).astype(int).clip(0, 1),
        "Z-score only (zscore_flag)": df["zscore_flag"].fillna(0).astype(int).clip(0, 1),
        "Final combined rule from aiops_log_analysis.py": final_combined,
    }

    rows = []
    for method_name, preds in methods.items():
        metrics = evaluate(y_ref, preds)
        rows.append(
            {
                "method": method_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    results = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 72)
    print("Evaluation of Anamoly Detection Methods")
    print("=" * 72)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
