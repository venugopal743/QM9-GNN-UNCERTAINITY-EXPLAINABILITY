"""Checks the API against the numbers published in the project's own result files."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["test_molecules"] == 13084


def test_models_match_repo_results():
    models = {m["experiment"]: m for m in client.get("/api/models").json()["models"]}
    assert set(models) == {"Original GCN", "Improved GCN", "GCN Comparison", "GraphSAGE", "GAT"}
    assert math.isclose(models["GCN Comparison"]["r2"], 0.933578, abs_tol=1e-6)
    assert math.isclose(models["GAT"]["mae"], 0.652288, abs_tol=1e-6)
    assert math.isclose(models["Original GCN"]["rmse"], 0.772918, abs_tol=1e-6)


def test_uncertainty_matches_repo_summary():
    u = client.get("/api/uncertainty").json()
    s = u["stats"]
    # From results/final/uncertainty_results.csv in the repo
    assert math.isclose(u["mc_mean_model"]["mae"], 0.274627, abs_tol=1e-4)
    assert math.isclose(s["pearson"], 0.21997, abs_tol=1e-4)
    assert math.isclose(s["low_uncertainty_mae"], 0.24349, abs_tol=1e-4)
    assert math.isclose(s["high_uncertainty_mae"], 0.305764, abs_tol=1e-4)
    assert len(u["bins"]) == 5
    assert sum(b["count"] for b in u["bins"]) == 13084
    # Coverage must be monotone in k
    cov = [c["observed"] for c in u["calibration"]]
    assert cov == sorted(cov)


def test_retention_is_monotone_ish_and_ends_at_full_mae():
    ret = client.get("/api/uncertainty").json()["retention"]
    assert ret[-1]["keep"] == 1.0
    assert math.isclose(ret[-1]["mae"], 0.274627, abs_tol=1e-4)
    assert ret[0]["mae"] < ret[-1]["mae"]


def test_scatter_and_errors():
    pts = client.get("/api/uncertainty/scatter?limit=500").json()["points"]
    assert len(pts) == 500
    err = client.get("/api/errors").json()
    assert sum(s["count"] for s in err["by_size"]) == 13084
    assert sum(err["histogram"]["counts"]) + err["histogram"]["outside_range"] == 13084


def test_training_history_and_overview():
    h = client.get("/api/training-history").json()
    assert set(h) == {"GCN", "GraphSAGE", "GAT", "Improved GCN"}
    assert all(len(v) == 10 for v in h.values())
    o = client.get("/api/overview").json()
    assert o["dataset"]["split"]["test"] == 13084
    assert o["best_model"]["name"] == "GCN Comparison"


def test_browse_filters_sorting_and_paging():
    r = client.get("/api/molecules?sort=absolute_error&order=desc&page_size=5").json()
    assert len(r["rows"]) == 5
    errs = [x["absolute_error"] for x in r["rows"]]
    assert errs == sorted(errs, reverse=True)
    small = client.get("/api/molecules?max_atoms=8&page_size=100").json()
    assert all(x["num_atoms"] <= 8 for x in small["rows"])
    assert small["total"] < 13084
    page2 = client.get("/api/molecules?page=2&page_size=5&sort=id&order=asc").json()
    assert [x["id"] for x in page2["rows"]] == [5, 6, 7, 8, 9]


def test_single_molecule_and_404():
    m = client.get("/api/molecules/0").json()
    assert m["num_atoms"] == 25
    assert math.isclose(m["target"], 8.533491, abs_tol=1e-5)
    assert math.isclose(m["interval_2sigma"][1] - m["interval_2sigma"][0], 4 * m["uncertainty"], abs_tol=1e-9)
    assert client.get("/api/molecules/99999").status_code == 404
    assert client.get("/api/molecules?sort=bogus").status_code == 422


def test_difficult():
    d = client.get("/api/difficult").json()["molecules"]
    assert len(d) == 20
    assert d[0]["difficulty_score"] >= d[-1]["difficulty_score"]
