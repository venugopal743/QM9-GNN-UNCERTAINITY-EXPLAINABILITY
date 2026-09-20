import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

INPUT_PATH = Path(
    "results/uncertainty/uncertainty_results.csv"
)

OUTPUT_PATH = Path(
    "results/uncertainty/uncertainty_vs_error.png"
)


df = pd.read_csv(INPUT_PATH)

uncertainty = df["uncertainty"]
absolute_error = df["absolute_error"]

correlation, p_value = pearsonr(
    uncertainty,
    absolute_error,
)

median_uncertainty = uncertainty.median()

low_uncertainty = df[
    uncertainty <= median_uncertainty
]

high_uncertainty = df[
    uncertainty > median_uncertainty
]

low_error = low_uncertainty[
    "absolute_error"
].mean()

high_error = high_uncertainty[
    "absolute_error"
].mean()


print("=" * 60)
print("UNCERTAINTY ANALYSIS")
print("=" * 60)

print(
    f"Pearson correlation: "
    f"{correlation:.6f}"
)

print(
    f"P-value: "
    f"{p_value:.6e}"
)

print(
    f"Median uncertainty: "
    f"{median_uncertainty:.6f}"
)

print(
    f"Low-uncertainty mean error: "
    f"{low_error:.6f}"
)

print(
    f"High-uncertainty mean error: "
    f"{high_error:.6f}"
)

print("=" * 60)


plt.figure(figsize=(8, 6))

plt.scatter(
    uncertainty,
    absolute_error,
    alpha=0.25,
    s=12,
)

plt.xlabel("Predictive Uncertainty")
plt.ylabel("Absolute Prediction Error")
plt.title(
    "MC Dropout Uncertainty vs Prediction Error"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
)

plt.close()

print(
    f"\nPlot saved to: {OUTPUT_PATH}"
)