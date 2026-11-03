"""Loads the project's result files once and derives the statistics the API serves.

Everything here comes from the CSV files exported by the QM9 GNN project
(see backend/data/). Nothing is hard-coded except descriptive text.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MODEL_DESCRIPTIONS = {
    "Original GCN": "2 GCN layers, 64 hidden units, no dropout or batch-norm, raw (unscaled) target.",
    "Improved GCN": "3 GCN layers, 128 hidden units, batch-norm, dropout 0.2, standardised target.",
    "GCN Comparison": "Same 3-layer GCN as Improved GCN, re-trained in the architecture comparison run.",
    "GraphSAGE": "3 SAGEConv layers, 128 hidden units, batch-norm, dropout 0.2.",
    "GAT": "3 GATConv layers (4 heads), 128 hidden units, batch-norm, ELU, dropout 0.2.",
}

HISTORY_FILES = {
    "GCN": "gcn_history.csv",
    "GraphSAGE": "graphsage_history.csv",
    "GAT": "gat_history.csv",
    "Improved GCN": "improved_training_history.csv",
}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


class Store:
    def __init__(self) -> None:
        self.models = _read("model_results.csv")
        self.summary = _read("final_results_summary.csv")
        self.mc_summary = _read("uncertainty_results.csv")
        self.histories = {k: _read(v) for k, v in HISTORY_FILES.items()}
        self.difficult = _read("difficult_molecules.csv")

        test = _read("test_set_analysis.csv").reset_index(names="id")
        self.test = test
        self._derive()

    # ------------------------------------------------------------------ derive
    def _derive(self) -> None:
        t = self.test
        err = t["prediction"] - t["target"]
        self.mc_metrics = {
            "mae": float(t["absolute_error"].mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "r2": float(1 - (err**2).sum() / ((t["target"] - t["target"].mean()) ** 2).sum()),
            "bias": float(err.mean()),
            "n": int(len(t)),
        }

        u, ae = t["uncertainty"], t["absolute_error"]
        corr = float(np.corrcoef(u, ae)[0, 1])
        spearman = float(u.rank().corr(ae.rank()))
        median_u = float(u.median())
        self.uncertainty_stats = {
            "pearson": corr,
            "spearman": spearman,
            "median_uncertainty": median_u,
            "mean_uncertainty": float(u.mean()),
            "max_uncertainty": float(u.max()),
            "low_uncertainty_mae": float(ae[u <= median_u].mean()),
            "high_uncertainty_mae": float(ae[u > median_u].mean()),
            "coverage_1sigma": float((ae <= u).mean()),
            "coverage_2sigma": float((ae <= 2 * u).mean()),
            "mc_samples": 50,
        }

        # Mean error per uncertainty quintile (does higher uncertainty mean higher error?)
        q = pd.qcut(u, 5, labels=False, duplicates="drop")
        g = t.groupby(q)
        self.uncertainty_bins = [
            {
                "bin": int(i) + 1,
                "mean_uncertainty": float(grp["uncertainty"].mean()),
                "mae": float(grp["absolute_error"].mean()),
                "count": int(len(grp)),
            }
            for i, grp in g
        ]

        # Calibration: expected vs observed coverage for a range of sigma multiples
        ks = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        self.calibration = [
            {"k": k, "observed": float((ae <= k * u).mean())} for k in ks
        ]

        # Selective prediction: keep only the most-confident X% of molecules
        order = np.argsort(u.to_numpy())
        sorted_ae = ae.to_numpy()[order]
        self.retention = [
            {
                "keep": frac,
                "mae": float(sorted_ae[: max(1, int(len(sorted_ae) * frac))].mean()),
            }
            for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        ]

        # Error by molecule size
        by_size = t.groupby("num_atoms")["absolute_error"].agg(["mean", "count"])
        self.error_by_size = [
            {"num_atoms": int(n), "mae": float(r["mean"]), "count": int(r["count"])}
            for n, r in by_size.iterrows()
        ]

        # Signed-error histogram
        counts, edges = np.histogram(err, bins=40, range=(-2.0, 2.0))
        self.error_hist = {
            "edges": edges.tolist(),
            "counts": counts.tolist(),
            "outside_range": int(((err < -2.0) | (err > 2.0)).sum()),
        }

        self.target_range = {
            "min": float(t["target"].min()),
            "max": float(t["target"].max()),
            "mean": float(t["target"].mean()),
            "std": float(t["target"].std()),
        }

    # ------------------------------------------------------------- query helpers
    def scatter(self, limit: int, seed: int = 42) -> list[dict]:
        n = len(self.test)
        limit = min(limit, n)
        idx = np.random.default_rng(seed).choice(n, size=limit, replace=False)
        rows = self.test.iloc[np.sort(idx)]
        return rows[["id", "target", "prediction", "uncertainty", "absolute_error"]].round(5).to_dict("records")

    def browse(
        self,
        sort: str,
        order: str,
        page: int,
        page_size: int,
        min_atoms: int | None,
        max_atoms: int | None,
        min_error: float | None,
        max_uncertainty: float | None,
    ) -> dict:
        df = self.test
        if min_atoms is not None:
            df = df[df["num_atoms"] >= min_atoms]
        if max_atoms is not None:
            df = df[df["num_atoms"] <= max_atoms]
        if min_error is not None:
            df = df[df["absolute_error"] >= min_error]
        if max_uncertainty is not None:
            df = df[df["uncertainty"] <= max_uncertainty]
        df = df.sort_values(sort, ascending=(order == "asc"), kind="stable")
        total = len(df)
        start = (page - 1) * page_size
        rows = df.iloc[start : start + page_size]
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": max(1, -(-total // page_size)),
            "rows": rows[["id", "num_atoms", "target", "prediction", "error", "absolute_error", "uncertainty"]]
            .round(5)
            .to_dict("records"),
        }

    def molecule(self, mol_id: int) -> dict | None:
        if mol_id < 0 or mol_id >= len(self.test):
            return None
        r = self.test.iloc[mol_id]
        u = float(r["uncertainty"])
        pred = float(r["prediction"])
        ae = float(r["absolute_error"])
        # Rank of this molecule's uncertainty / error among all test molecules
        u_rank = float((self.test["uncertainty"] < u).mean())
        e_rank = float((self.test["absolute_error"] < ae).mean())
        return {
            "id": int(r["id"]),
            "num_atoms": int(r["num_atoms"]),
            "target": float(r["target"]),
            "prediction": pred,
            "signed_error": float(r["error"]),
            "absolute_error": ae,
            "uncertainty": u,
            "interval_1sigma": [pred - u, pred + u],
            "interval_2sigma": [pred - 2 * u, pred + 2 * u],
            "within_1sigma": bool(ae <= u),
            "within_2sigma": bool(ae <= 2 * u),
            "uncertainty_percentile": u_rank * 100,
            "error_percentile": e_rank * 100,
        }


store = Store()
