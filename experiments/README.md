# Experiments

This directory contains experiment configurations, metrics, and comparisons for the QM9 molecular property prediction project.

## Target

Primary prediction target:

**HOMO-LUMO gap (gap)**

QM9 target index:

4

## Dataset Split

All experiments use the same reproducible split:

- Training: 104,664 molecules
- Validation: 13,083 molecules
- Test: 13,084 molecules

Random seed:

42

## Model Comparison

| Model | Test MSE | Test RMSE | Test MAE | R² |
|---|---:|---:|---:|---:|
| Original GCN | 0.597403 | 0.772918 | 0.605839 | 0.641363 |
| Improved GCN | 0.126668 | 0.355905 | 0.265042 | 0.923957 |

## Original GCN

- 2 GCN layers
- 64 hidden dimensions
- Global mean pooling
- Fully connected prediction head
- No dropout
- No batch normalization
- Raw target values

## Improved GCN

- 3 GCN layers
- 128 hidden dimensions
- Batch normalization
- Dropout
- Global mean pooling
- Fully connected prediction head
- Standardized target values during training

Predictions are converted back to the original target units before final test evaluation.

## Next Experiments

The next stage will compare:

- GCN
- GraphSAGE
- GAT

The goal is to evaluate how different graph neural network architectures affect molecular property prediction performance.
