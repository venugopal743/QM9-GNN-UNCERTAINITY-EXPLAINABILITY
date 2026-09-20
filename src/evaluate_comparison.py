import csv
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from dataset import QM9GapDataset
from split_data import split_dataset
from target_scaler import TargetScaler
from models_comparison import (
    GCNModel,
    GraphSAGEModel,
    GATModel,
)


BATCH_SIZE = 64
SEED = 42

MODEL_DIR = Path("models/comparison")
RESULTS_DIR = Path("results/comparison")

MODEL_CONFIGS = {
    "GCN": (
        GCNModel,
        MODEL_DIR / "gcn_gap.pt",
    ),
    "GraphSAGE": (
        GraphSAGEModel,
        MODEL_DIR / "graphsage_gap.pt",
    ),
    "GAT": (
        GATModel,
        MODEL_DIR / "gat_gap.pt",
    ),
}


def evaluate_model(
    model,
    test_loader,
    scaler,
    device,
):
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

            output = scaler.inverse_transform(
                normalized_output
            )

            target = batch.y.view(-1)

            predictions.extend(
                output.cpu().numpy()
            )

            targets.extend(
                target.cpu().numpy()
            )

    mse = mean_squared_error(
        targets,
        predictions,
    )

    rmse = mse ** 0.5

    mae = mean_absolute_error(
        targets,
        predictions,
    )

    r2 = r2_score(
        targets,
        predictions,
    )

    return mse, rmse, mae, r2


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

    print(
        f"Test molecules: "
        f"{len(test_dataset)}"
    )

    scaler = TargetScaler()
    scaler.fit(train_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    results = []

    for model_name, (
        model_class,
        model_path,
    ) in MODEL_CONFIGS.items():

        print()
        print("=" * 60)
        print(f"Evaluating {model_name}")
        print("=" * 60)

        model = model_class().to(device)

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
            )
        )

        mse, rmse, mae, r2 = evaluate_model(
            model,
            test_loader,
            scaler,
            device,
        )

        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"R²:   {r2:.6f}")

        results.append(
            {
                "model": model_name,
                "test_mse": mse,
                "test_rmse": rmse,
                "test_mae": mae,
                "r2": r2,
            }
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "model_comparison.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "test_mse",
                "test_rmse",
                "test_mae",
                "r2",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    for result in results:

        print(
            f"{result['model']:10s} | "
            f"MSE: {result['test_mse']:.6f} | "
            f"RMSE: {result['test_rmse']:.6f} | "
            f"MAE: {result['test_mae']:.6f} | "
            f"R²: {result['r2']:.6f}"
        )

    print()
    print(
        f"Results saved to: {output_path}"
    )