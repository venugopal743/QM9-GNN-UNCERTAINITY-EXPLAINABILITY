import torch
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw

from dataset import QM9GapDataset
from split_data import split_dataset
from target_scaler import TargetScaler
from model import GCNModel


SEED = 42
MODEL_PATH = "models/gcn_gap.pt"

OUTPUT_PATH = Path(
    "results/explainability/explained_molecule.png"
)


def enable_gradients(model, molecule, scaler, device):
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
    ).cpu().numpy()

    prediction = scaler.inverse_transform(
        normalized_prediction.detach().cpu()
    ).item()

    return prediction, importance


def molecule_to_rdkit(molecule):
    """
    Reconstruct an RDKit molecule from the QM9
    atomic numbers and graph connectivity.
    """

    rdkit_mol = Chem.RWMol()

    atom_map = {}

    for index, atomic_number in enumerate(
        molecule.z.tolist()
    ):
        atom = Chem.Atom(int(atomic_number))
        rdkit_index = rdkit_mol.AddAtom(atom)
        atom_map[index] = rdkit_index

    added_bonds = set()

    edge_index = molecule.edge_index.numpy()

    for source, target in zip(
        edge_index[0],
        edge_index[1],
    ):
        source = int(source)
        target = int(target)

        if source == target:
            continue

        bond_key = tuple(
            sorted((source, target))
        )

        if bond_key in added_bonds:
            continue

        try:
            rdkit_mol.AddBond(
                atom_map[source],
                atom_map[target],
                Chem.BondType.SINGLE,
            )

            added_bonds.add(bond_key)

        except Exception:
            pass

    mol = rdkit_mol.GetMol()

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    return mol


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

    molecule = test_dataset[0]

    prediction, importance = enable_gradients(
        model,
        molecule,
        scaler,
        device,
    )

    actual = molecule.y.item()

    print()
    print("=" * 60)
    print("MOLECULAR EXPLAINABILITY")
    print("=" * 60)

    print(
        f"Number of atoms: "
        f"{molecule.num_nodes}"
    )

    print(
        f"Actual gap: "
        f"{actual:.6f}"
    )

    print(
        f"Predicted gap: "
        f"{prediction:.6f}"
    )

    print()
    print("Most important atoms:")

    ranked_atoms = np.argsort(
        importance
    )[::-1]

    for rank, atom_index in enumerate(
        ranked_atoms[:10],
        start=1,
    ):
        atomic_number = int(
            molecule.z[atom_index].item()
        )

        print(
            f"{rank:02d}. Atom {atom_index:02d} "
            f"(Z={atomic_number}) "
            f"importance={importance[atom_index]:.8f}"
        )

    mol = molecule_to_rdkit(molecule)

    # Normalize importance scores to 0–1.
    max_importance = importance.max()

    if max_importance > 0:
        normalized_importance = (
            importance / max_importance
        )
    else:
        normalized_importance = importance

    atom_colors = {}

    for atom_index, score in enumerate(
        normalized_importance
    ):
        intensity = float(score)

        # Highlight important atoms using
        # an intensity-based overlay.
        atom_colors[atom_index] = (
            1.0,
            1.0 - intensity,
            1.0 - intensity,
        )

    highlight_atoms = [
        int(index)
        for index in ranked_atoms[:10]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Draw.MolToImage(
        mol,
        size=(900, 700),
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
    )

    image.save(OUTPUT_PATH)

    print()
    print(
        f"Explained molecule saved to: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 60)