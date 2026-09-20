import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNModel(nn.Module):
    """Improved GCN for QM9 HOMO-LUMO gap prediction."""

    def __init__(
        self,
        input_dim=11,
        hidden_dim=128,
        output_dim=1,
        dropout=0.2,
    ):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(
            hidden_dim,
            output_dim,
        )

        self.dropout = dropout

    def forward(self, x, edge_index, batch):

        # GCN layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # GCN layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # GCN layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Molecular-level representation
        x = global_mean_pool(
            x,
            batch,
        )

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.fc2(x)

        return x