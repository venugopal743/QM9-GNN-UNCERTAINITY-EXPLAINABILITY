import torch


def evaluate(model, loader, criterion, device):
    """Evaluate the model on a dataset."""

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

            targets = batch.y.view(-1)

            squared_error = torch.sum(
                (predictions - targets) ** 2
            )

            absolute_error = torch.sum(
                torch.abs(predictions - targets)
            )

            total_squared_error += squared_error.item()
            total_absolute_error += absolute_error.item()
            total_graphs += batch.num_graphs

    mse = total_squared_error / total_graphs
    mae = total_absolute_error / total_graphs

    return mse, mae