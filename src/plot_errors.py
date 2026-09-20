import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.loader import DataLoader

from dataset import QM9GapDataset
from split_data import split_dataset
from model import GCNModel
from target_scaler import TargetScaler

BATCH_SIZE = 64
SEED = 42
MODEL_PATH = "models/gcn_gap.pt"

OUTPUT_PATH = Path("results/improved_gcn/error_distribution.png")

torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = QM9GapDataset()
train_dataset, _, test_dataset = split_dataset(dataset)

scaler = TargetScaler()
scaler.fit(train_dataset)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = GCNModel().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
model.eval()

errors = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)

        normalized_output = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        ).view(-1)

        predictions = scaler.inverse_transform(normalized_output)
        targets = batch.y.view(-1)

        batch_errors = (
            predictions - targets
        ).cpu().numpy()

        errors.extend(batch_errors)

plt.figure(figsize=(8, 5))

plt.hist(
    errors,
    bins=50,
    alpha=0.75,
)

plt.axvline(
    0,
    linestyle="--",
    label="Zero error",
)

plt.xlabel("Prediction Error")
plt.ylabel("Number of Molecules")
plt.title("Improved GCN: Prediction Error Distribution")
plt.legend()
plt.grid(True, alpha=0.3)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
)

plt.close()

print(f"Plot saved to: {OUTPUT_PATH}")