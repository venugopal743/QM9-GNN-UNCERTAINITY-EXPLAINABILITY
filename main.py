"""QM9 · GraphMol Lab — API + static frontend.

Run from the backend/ folder:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data import MODEL_DESCRIPTIONS, store

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(
    title="QM9 · GraphMol Lab API",
    description="Results of GNN HOMO-LUMO gap prediction on QM9 with MC-Dropout uncertainty and gradient explainability.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "test_molecules": len(store.test)}


@app.get("/api/overview")
def overview() -> dict:
    summary = dict(zip(store.summary["component"], store.summary["result"]))
    best = store.models.sort_values("r2", ascending=False).iloc[0]
    return {
        "dataset": {
            "name": "QM9",
            "molecules": 130_831,
            "atom_features": 11,
            "target_properties": 19,
            "target": "HOMO-LUMO gap",
            "target_index": 4,
            "unit": "eV",
            "split": {"train": 104_664, "validation": 13_083, "test": 13_084},
            "seed": 42,
        },
        "target_distribution": store.target_range,
        "best_model": {
            "name": best["experiment"],
            "r2": float(best["r2"]),
            "mae": float(best["mae"]),
            "rmse": float(best["rmse"]),
        },
        "summary": summary,
    }


@app.get("/api/models")
def models() -> dict:
    rows = store.models.round(6).to_dict("records")
    for r in rows:
        r["description"] = MODEL_DESCRIPTIONS.get(r["experiment"], "")
    return {"models": rows}


@app.get("/api/training-history")
def training_history() -> dict:
    return {name: df.round(6).to_dict("records") for name, df in store.histories.items()}


@app.get("/api/uncertainty")
def uncertainty() -> dict:
    return {
        "stats": store.uncertainty_stats,
        "mc_mean_model": store.mc_metrics,
        "bins": store.uncertainty_bins,
        "calibration": store.calibration,
        "retention": store.retention,
    }


@app.get("/api/uncertainty/scatter")
def uncertainty_scatter(limit: int = Query(2000, ge=100, le=13084)) -> dict:
    return {"points": store.scatter(limit)}


@app.get("/api/errors")
def errors() -> dict:
    return {
        "by_size": store.error_by_size,
        "histogram": store.error_hist,
        "mc_mean_model": store.mc_metrics,
    }


@app.get("/api/difficult")
def difficult() -> dict:
    return {"molecules": store.difficult.round(5).to_dict("records")}


SortKey = Literal["id", "num_atoms", "target", "prediction", "absolute_error", "uncertainty", "error"]


@app.get("/api/molecules")
def molecules(
    sort: SortKey = "absolute_error",
    order: Literal["asc", "desc"] = "desc",
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=100),
    min_atoms: int | None = Query(None, ge=1, le=30),
    max_atoms: int | None = Query(None, ge=1, le=30),
    min_error: float | None = Query(None, ge=0),
    max_uncertainty: float | None = Query(None, ge=0),
) -> dict:
    return store.browse(sort, order, page, page_size, min_atoms, max_atoms, min_error, max_uncertainty)


@app.get("/api/molecules/{mol_id}")
def molecule(mol_id: int) -> dict:
    result = store.molecule(mol_id)
    if result is None:
        raise HTTPException(404, f"Molecule id must be between 0 and {len(store.test) - 1}")
    return result


# ---- frontend (mounted last so /api/* wins) ---------------------------------
app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")
app.mount("/vendor", StaticFiles(directory=FRONTEND_DIR / "vendor"), name="vendor")
app.mount("/img", StaticFiles(directory=FRONTEND_DIR / "img"), name="img")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
