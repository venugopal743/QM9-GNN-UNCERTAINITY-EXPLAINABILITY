import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("results/final")

OUTPUT_PATH = OUTPUT_DIR / "final_results_summary.csv"


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_results = pd.DataFrame(
        [
            {
                "experiment": "Original GCN",
                "mse": 0.597403,
                "rmse": 0.772918,
                "mae": 0.605839,
                "r2": 0.641363,
            },
            {
                "experiment": "Improved GCN",
                "mse": 0.126668,
                "rmse": 0.355905,
                "mae": 0.265042,
                "r2": 0.923957,
            },
            {
                "experiment": "GCN Comparison",
                "mse": 0.110643,
                "rmse": 0.332630,
                "mae": 0.248973,
                "r2": 0.933578,
            },
            {
                "experiment": "GraphSAGE",
                "mse": 0.121099,
                "rmse": 0.347992,
                "mae": 0.255409,
                "r2": 0.927301,
            },
            {
                "experiment": "GAT",
                "mse": 0.686732,
                "rmse": 0.828693,
                "mae": 0.652288,
                "r2": 0.587736,
            },
        ]
    )

    uncertainty_results = pd.DataFrame(
        [
            {
                "experiment": "MC Dropout",
                "mean_absolute_error": 0.274627,
                "uncertainty_error_correlation": 0.219970,
                "low_uncertainty_mae": 0.243490,
                "high_uncertainty_mae": 0.305764,
            }
        ]
    )

    model_results.to_csv(
        OUTPUT_DIR / "model_results.csv",
        index=False,
    )

    uncertainty_results.to_csv(
        OUTPUT_DIR / "uncertainty_results.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "component": "Dataset",
                "result": "QM9",
            },
            {
                "component": "Target",
                "result": "HOMO-LUMO gap",
            },
            {
                "component": "Dataset split",
                "result": "80% train / 10% validation / 10% test",
            },
            {
                "component": "Random seed",
                "result": "42",
            },
            {
                "component": "Test molecules",
                "result": "13,084",
            },
            {
                "component": "Best architecture experiment",
                "result": "GCN comparison run",
            },
            {
                "component": "Best comparison R2",
                "result": "0.933578",
            },
            {
                "component": "MC Dropout correlation",
                "result": "0.219970",
            },
            {
                "component": "Low uncertainty MAE",
                "result": "0.243490",
            },
            {
                "component": "High uncertainty MAE",
                "result": "0.305764",
            },
            {
                "component": "Explainability",
                "result": "Gradient-based atom importance",
            },
        ]
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 60)
    print("FINAL EXPERIMENT SUMMARY CREATED")
    print("=" * 60)

    print(
        f"Model results: "
        f"{OUTPUT_DIR / 'model_results.csv'}"
    )

    print(
        f"Uncertainty results: "
        f"{OUTPUT_DIR / 'uncertainty_results.csv'}"
    )

    print(
        f"Summary: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 60)