import pandas as pd
from pathlib import Path


INPUT_PATH = Path(
    "results/analysis/test_set_analysis.csv"
)

OUTPUT_DIR = Path(
    "results/analysis"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "difficult_molecules.csv"
)


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("DIFFICULT MOLECULE ANALYSIS")
    print("=" * 60)

    print("\nTop 10 molecules by absolute error:")

    highest_error = (
        df.sort_values(
            "absolute_error",
            ascending=False,
        )
        .head(10)
    )

    print(
        highest_error[
            [
                "target",
                "prediction",
                "absolute_error",
                "uncertainty",
                "num_atoms",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nTop 10 molecules by uncertainty:")

    highest_uncertainty = (
        df.sort_values(
            "uncertainty",
            ascending=False,
        )
        .head(10)
    )

    print(
        highest_uncertainty[
            [
                "target",
                "prediction",
                "absolute_error",
                "uncertainty",
                "num_atoms",
            ]
        ].to_string(
            index=False
        )
    )

    # Normalize error and uncertainty
    # before combining them.
    error_norm = (
        df["absolute_error"]
        / df["absolute_error"].max()
    )

    uncertainty_norm = (
        df["uncertainty"]
        / df["uncertainty"].max()
    )

    df["difficulty_score"] = (
        0.5 * error_norm
        + 0.5 * uncertainty_norm
    )

    difficult = (
        df.sort_values(
            "difficulty_score",
            ascending=False,
        )
        .head(20)
    )

    difficult.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nTop 10 difficult molecules:")

    print(
        difficult[
            [
                "target",
                "prediction",
                "absolute_error",
                "uncertainty",
                "num_atoms",
                "difficulty_score",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print("=" * 60)

    print(
        f"\nSaved difficult molecules to: "
        f"{OUTPUT_PATH}"
    )