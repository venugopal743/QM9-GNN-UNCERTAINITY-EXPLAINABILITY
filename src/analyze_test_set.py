import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import QM9GapDataset
from split_data import split_dataset


INPUT_PATH = Path(
    "results/uncertainty/uncertainty_results.csv"
)

OUTPUT_DIR = Path(
    "results/analysis"
)

OUTPUT_CSV = OUTPUT_DIR / "test_set_analysis.csv"
OUTPUT_PLOT = OUTPUT_DIR / "error_vs_molecule_size.png"


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading QM9 dataset...")

    dataset = QM9GapDataset()

    _, _, test_dataset = split_dataset(dataset)

    print(
        f"Test molecules: {len(test_dataset)}"
    )

    results = pd.read_csv(INPUT_PATH)

    atom_counts = []

    for index in range(len(test_dataset)):
        molecule = test_dataset[index]
        atom_counts.append(molecule.num_nodes)

    results["num_atoms"] = atom_counts

    results["error"] = (
        results["prediction"]
        - results["target"]
    )

    results["absolute_error"] = (
        results["error"].abs()
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    correlation = results[
        "num_atoms"
    ].corr(
        results["absolute_error"]
    )

    print()
    print("=" * 60)
    print("TEST SET SCIENTIFIC ANALYSIS")
    print("=" * 60)

    print(
        f"Mean absolute error: "
        f"{results['absolute_error'].mean():.6f}"
    )

    print(
        f"Mean molecule size: "
        f"{results['num_atoms'].mean():.2f} atoms"
    )

    print(
        f"Smallest molecule: "
        f"{results['num_atoms'].min()} atoms"
    )

    print(
        f"Largest molecule: "
        f"{results['num_atoms'].max()} atoms"
    )

    print(
        f"Correlation between molecule size "
        f"and absolute error: "
        f"{correlation:.6f}"
    )

    print("=" * 60)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        results["num_atoms"],
        results["absolute_error"],
        alpha=0.25,
        s=12,
    )

    plt.xlabel(
        "Number of Atoms"
    )

    plt.ylabel(
        "Absolute Prediction Error"
    )

    plt.title(
        "Prediction Error vs Molecular Size"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=300,
    )

    plt.close()

    print(
        f"\nAnalysis data saved to: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Plot saved to: "
        f"{OUTPUT_PLOT}"
    )