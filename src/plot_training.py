import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

HISTORY_PATH = Path("results/improved_gcn/improved_training_history.csv")
OUTPUT_PATH = Path("results/improved_gcn/training_validation_mse.png")

history = pd.read_csv(HISTORY_PATH)

plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["train_mse"],
    marker="o",
    label="Training MSE",
)

plt.plot(
    history["epoch"],
    history["val_mse"],
    marker="o",
    label="Validation MSE",
)

plt.xlabel("Epoch")
plt.ylabel("Normalized MSE")
plt.title("Improved GCN Training and Validation MSE")
plt.legend()
plt.grid(True, alpha=0.3)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print(f"Plot saved to: {OUTPUT_PATH}")