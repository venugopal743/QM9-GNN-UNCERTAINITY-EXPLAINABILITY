import torch


class TargetScaler:
    """
    Standardizes the HOMO-LUMO gap target.

    Formula:
        z = (y - mean) / std

    Predictions can be converted back using:
        y = z * std + mean
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset):
        """Calculate mean and standard deviation from training data only."""

        targets = []

        for index in range(len(dataset)):
            targets.append(
                dataset[index].y.item()
            )

        targets = torch.tensor(
            targets,
            dtype=torch.float32,
        )

        self.mean = targets.mean()
        self.std = targets.std()

    def transform(self, target):
        """Normalize target values."""

        return (
            target - self.mean
        ) / self.std

    def inverse_transform(self, target):
        """Convert normalized values back to original units."""

        return (
            target * self.std
        ) + self.mean