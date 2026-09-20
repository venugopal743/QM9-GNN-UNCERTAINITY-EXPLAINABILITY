import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.loader import DataLoader

from dataset import QM9GapDataset
from split_data import split_dataset
from target_scaler import TargetScaler
from model import GCNModel


SEED = 42
MODEL_PATH = "models/gcn_gap.pt"
OUTPUT_PATH = Path(
    "results/explainability/atom_importance.png"
)


def compute_atom_importance(
    model,
    molecule,
    scaler,
    device,
):
    """
    Compute gradient-based importance for each atom.

    Importance is based on the magnitude of the
    gradient of the predicted target with respect
    to the atom feature representation.
    """

    model.eval()

    x = molecule.x.clone().detach()
    x.requires_grad_(True)

    edge_index = molecule.edge_index

    batch = torch.zeros(
        molecule.num_nodes,
        dtype=torch.long,
        device=device,
    )

    x = x.to(device)
    edge_index = edge_index.to(device)

    batch = batch.to(device)

    model.zero_grad()

    normalized_prediction = model(
        x,
        edge_index,
        batch,
    ).view(-1)[0]

    normalized_prediction.backward()

    gradients = x.grad.detach()

    importance = torch.norm(
        gradients,
        p=2,
        dim=1,
    )

    importance = importance.cpu().numpy()

    prediction = scaler.inverse_transform(
        normalized_prediction.detach().cpu()
    ).item()

    return prediction, importance


if __name__ == "__main__":

    torch.manual_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    dataset = QM9GapDataset()

    train_dataset, _, test_dataset = (
        split_dataset(dataset)
    )

    scaler = TargetScaler()
    scaler.fit(train_dataset)

    model = GCNModel().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    # Select a test molecule.
    molecule = test_dataset[0]

    prediction, importance = (
        compute_atom_importance(
            model,
            molecule,
            scaler,
            device,
        )
    )

    actual = molecule.y.item()

    atom_indices = np.arange(
        molecule.num_nodes
    )

    print()
    print("=" * 60)
    print("GRAPH EXPLAINABILITY")
    print("=" * 60)

    print(
        f"Number of atoms: "
        f"{molecule.num_nodes}"
    )

    print(
        f"Actual HOMO-LUMO gap: "
        f"{actual:.6f}"
    )

    print(
        f"Predicted HOMO-LUMO gap: "
        f"{prediction:.6f}"
    )

    print()
    print("Atom importance scores:")

    for index, score in zip(
        atom_indices,
        importance,
    ):
        print(
            f"Atom {index:02d}: "
            f"{score:.8f}"
        )

    print("=" * 60)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        atom_indices,
        importance,
    )

    plt.xlabel("Atom Index")
    plt.ylabel("Gradient Importance")
    plt.title(
        "Gradient-Based Atom Importance"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
    )

    plt.close()

    print(
        f"\nExplanation plot saved to: "
        f"{OUTPUT_PATH}"
    )