import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader

from dataset import QM9GapDataset
from split_data import split_dataset
from target_scaler import TargetScaler
from model import GCNModel


BATCH_SIZE = 64
MC_SAMPLES = 50
SEED = 42

MODEL_PATH = "models/gcn_gap.pt"

RESULTS_DIR = Path("results/uncertainty")
PREDICTIONS_PATH = RESULTS_DIR / "mc_predictions.npy"
CSV_PATH = RESULTS_DIR / "uncertainty_results.csv"


def enable_dropout(model):
    """Enable dropout layers during inference."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(
    model,
    loader,
    scaler,
    device,
    mc_samples=50,
):
    """Generate stochastic predictions using MC Dropout."""

    predictions = []

    for sample in range(mc_samples):

        enable_dropout(model)

        sample_predictions = []

        with torch.no_grad():

            for batch in loader:

                batch = batch.to(device)

                normalized_output = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                ).view(-1)

                output = scaler.inverse_transform(
                    normalized_output
                )

                sample_predictions.append(
                    output.cpu().numpy()
                )

        predictions.append(
            np.concatenate(sample_predictions)
        )

        if (sample + 1) % 10 == 0:
            print(
                f"Completed MC sample "
                f"{sample + 1}/{mc_samples}"
            )

    return np.stack(predictions)


if __name__ == "__main__":

    torch.manual_seed(SEED)
    np.random.seed(SEED)

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

    model = GCNModel().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    print()
    print(
        f"Running {MC_SAMPLES} "
        "Monte Carlo samples..."
    )

    mc_predictions = mc_dropout_predict(
        model,
        test_loader,
        scaler,
        device,
        MC_SAMPLES,
    )

    np.save(
        PREDICTIONS_PATH,
        mc_predictions,
    )

    mean_prediction = mc_predictions.mean(
        axis=0
    )

    uncertainty = mc_predictions.std(
        axis=0
    )

    targets = []

    for batch in test_loader:
        targets.extend(
            batch.y.numpy()
        )

    targets = np.asarray(targets)

    absolute_error = np.abs(
        mean_prediction - targets
    )

    results = pd.DataFrame(
        {
            "target": targets,
            "prediction": mean_prediction,
            "uncertainty": uncertainty,
            "absolute_error": absolute_error,
        }
    )

    results.to_csv(
        CSV_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("MONTE CARLO DROPOUT RESULTS")
    print("=" * 60)

    print(
        f"Mean prediction: "
        f"{mean_prediction.mean():.6f}"
    )

    print(
        f"Mean uncertainty: "
        f"{uncertainty.mean():.6f}"
    )

    print(
        f"Maximum uncertainty: "
        f"{uncertainty.max():.6f}"
    )

    print(
        f"Mean absolute error: "
        f"{absolute_error.mean():.6f}"
    )

    print(
        f"Prediction standard deviation: "
        f"{mean_prediction.std():.6f}"
    )

    print("=" * 60)

    print(
        f"\nPredictions saved to: "
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"Results saved to: "
        f"{CSV_PATH}"
    )