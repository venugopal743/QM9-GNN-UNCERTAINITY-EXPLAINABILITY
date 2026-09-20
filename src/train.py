import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from dataset import QM9GapDataset
from split_data import split_dataset
from model import GCNModel
from evaluate import evaluate
from target_scaler import TargetScaler


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
SEED = 42

MODEL_PATH = "models/gcn_gap.pt"
HISTORY_PATH = "results/training_history.csv"


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
):
    """Train the model for one epoch."""

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
            loss.item()
            * batch.num_graphs
        )

    return total_loss / len(loader.dataset)


def evaluate_normalized(
    model,
    loader,
    criterion,
    device,
    scaler,
):
    """Evaluate using normalized targets."""

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
                torch.abs(
                    predictions - targets
                )
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


if __name__ == "__main__":

    torch.manual_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # Load dataset
    dataset = QM9GapDataset()

    print(
        f"Total molecules: {len(dataset)}"
    )

    # Reproducible 80/10/10 split
    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = split_dataset(dataset)

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

    # --------------------------------------------------
    # Fit target scaler ONLY on training data
    # --------------------------------------------------

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

    # Data loaders
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

    # Model
    model = GCNModel().to(device)

    criterion = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_mse = float("inf")

    history = []

    print(
        "\nStarting improved GCN training...\n"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
        )

        val_mse, val_mae = (
            evaluate_normalized(
                model,
                val_loader,
                criterion,
                device,
                scaler,
            )
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
                MODEL_PATH,
            )

            print(
                f"  -> Saved best model "
                f"(Val MSE: "
                f"{val_mse:.6f})"
            )

    # Save training history
    Path("results").mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        HISTORY_PATH,
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
        "\nTraining complete!"
    )

    print(
        f"Best validation MSE: "
        f"{best_val_mse:.6f}"
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Training history saved to: "
        f"{HISTORY_PATH}"
    )