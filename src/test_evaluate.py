import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from dataset import QM9GapDataset
from split_data import split_dataset
from model import GCNModel
from target_scaler import TargetScaler


BATCH_SIZE = 64
SEED = 42
MODEL_PATH = "models/gcn_gap.pt"


if __name__ == "__main__":

    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load dataset
    dataset = QM9GapDataset()

    # Reproduce the exact same split
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset
    )

    print(f"Test molecules: {len(test_dataset)}")

    # Fit scaler ONLY on training data
    scaler = TargetScaler()
    scaler.fit(train_dataset)

    print(f"Target mean: {scaler.mean.item():.6f}")
    print(f"Target std: {scaler.std.item():.6f}")

    # Test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Create model
    model = GCNModel().to(device)

    # Load best model
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in test_loader:

            batch = batch.to(device)

            # Model predicts normalized target
            normalized_output = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            ).view(-1)

            # Convert prediction back to original units
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

    # Calculate metrics in original units
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

    print("\n==============================")
    print("IMPROVED GCN TEST RESULTS")
    print("==============================")

    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")

    print("==============================")