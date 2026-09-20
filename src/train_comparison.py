import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from dataset import QM9GapDataset
from split_data import split_dataset
from target_scaler import TargetScaler
from models_comparison import GCNModel, GraphSAGEModel, GATModel


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
SEED = 42

RESULTS_DIR = Path("results/comparison")
MODELS_DIR = Path("models/comparison")


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
):
    model.train()

    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        predictions = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        ).view(-1)

        targets = scaler.transform(
            batch.y.view(-1)
        )

        loss = criterion(
            predictions,
            targets,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * batch.num_graphs
        )

    return total_loss / len(loader.dataset)


def evaluate(
    model,
    loader,
    criterion,
    device,
    scaler,
):
    model.eval()

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_graphs = 0

    with torch.no_grad():

        for batch in loader:
            batch = batch.to(device)

            predictions = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            ).view(-1)

            targets = scaler.transform(
                batch.y.view(-1)
            )

            squared_error = torch.sum(
                (predictions - targets) ** 2
            )

            absolute_error = torch.sum(
                torch.abs(predictions - targets)
            )

            total_squared_error += (
                squared_error.item()
            )

            total_absolute_error += (
                absolute_error.item()
            )

            total_graphs += batch.num_graphs

    mse = (
        total_squared_error
        / total_graphs
    )

    mae = (
        total_absolute_error
        / total_graphs
    )

    return mse, mae


def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    device,
    scaler,
):
    criterion = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_mse = float("inf")

    history = []

    model_path = (
        MODELS_DIR
        / f"{model_name.lower()}_gap.pt"
    )

    print()
    print("=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        train_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
        )

        val_mse, val_mae = evaluate(
            model,
            val_loader,
            criterion,
            device,
            scaler,
        )

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "val_mae": val_mae,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train MSE: {train_mse:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )

        if val_mse < best_val_mse:

            best_val_mse = val_mse

            torch.save(
                model.state_dict(),
                model_path,
            )

            print(
                f"  -> Saved best model "
                f"(Val MSE: {val_mse:.6f})"
            )

    history_path = (
        RESULTS_DIR
        / f"{model_name.lower()}_history.csv"
    )

    with open(
        history_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_mse",
                "val_mse",
                "val_mae",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    print(
        f"\nBest validation MSE: "
        f"{best_val_mse:.6f}"
    )

    return best_val_mse


if __name__ == "__main__":

    torch.manual_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = QM9GapDataset()

    train_dataset, val_dataset, test_dataset = (
        split_dataset(dataset)
    )

    print(
        f"Training molecules: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation molecules: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test molecules: "
        f"{len(test_dataset)}"
    )

    scaler = TargetScaler()

    scaler.fit(train_dataset)

    print(
        f"\nTarget mean: "
        f"{scaler.mean.item():.6f}"
    )

    print(
        f"Target std: "
        f"{scaler.std.item():.6f}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    models = {
        "GCN": GCNModel(),
        "GraphSAGE": GraphSAGEModel(),
        "GAT": GATModel(),
    }

    for model_name, model in models.items():

        model = model.to(device)

        train_model(
            model_name,
            model,
            train_loader,
            val_loader,
            device,
            scaler,
        )

    print()
    print("=" * 60)
    print("ARCHITECTURE COMPARISON TRAINING COMPLETE")
    print("=" * 60)