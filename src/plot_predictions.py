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

OUTPUT_PATH = Path("results/improved_gcn/actual_vs_predicted.png")

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

predictions = []
targets = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)

        normalized_output = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        ).view(-1)

        output = scaler.inverse_transform(normalized_output)
        target = batch.y.view(-1)

        predictions.extend(output.cpu().numpy())
        targets.extend(target.cpu().numpy())

plt.figure(figsize=(7, 7))

plt.scatter(
    targets,
    predictions,
    alpha=0.25,
    s=12,
)

minimum = min(min(targets), min(predictions))
maximum = max(max(targets), max(predictions))

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect prediction",
)

plt.xlabel("Actual HOMO-LUMO Gap")
plt.ylabel("Predicted HOMO-LUMO Gap")
plt.title("Improved GCN: Actual vs Predicted")
plt.legend()
plt.grid(True, alpha=0.3)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
)

plt.close()

print(f"Plot saved to: {OUTPUT_PATH}")