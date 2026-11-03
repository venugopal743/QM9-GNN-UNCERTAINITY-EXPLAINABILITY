/* Static replacement for the FastAPI backend, used on GitHub Pages.
   Serves the same response shapes as /api/* from pre-built JSON files in data/.
   The molecule browser/filter logic is a port of backend/app/data.py. */
(() => {
  const FILES = {
    "/api/overview": "data/overview.json",
    "/api/models": "data/models.json",
    "/api/training-history": "data/training-history.json",
    "/api/uncertainty": "data/uncertainty.json",
    "/api/uncertainty/scatter": "data/scatter.json",
    "/api/errors": "data/errors.json",
    "/api/difficult": "data/difficult.json",
  };

  const getJson = async (file) => {
    const r = await fetch(file); // relative URL, so it works under /<repo-name>/
    if (!r.ok) throw new Error(`${file} → ${r.status}`);
    return r.json();
  };

  let dbPromise = null;
  const db = () => (dbPromise ||= getJson("data/molecules.json").then(build));

  function build(c) {
    const n = c.target.length;
    const rows = new Array(n);
    for (let i = 0; i < n; i++) {
      const t = c.target[i], p = c.prediction[i], err = p - t;
      rows[i] = { id: i, num_atoms: c.num_atoms[i], target: t, prediction: p, error: err,
                  absolute_error: Math.abs(err), uncertainty: c.uncertainty[i] };
    }
    const sortedU = Float64Array.from(rows, (r) => r.uncertainty).sort();
    const sortedE = Float64Array.from(rows, (r) => r.absolute_error).sort();
    return { rows, sortedU, sortedE };
  }

  // fraction of values strictly below v in an ascending-sorted array
  const fracBelow = (sorted, v) => {
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < v) lo = mid + 1; else hi = mid; }
    return lo / sorted.length;
  };

  async function browse(q) {
    const { rows } = await db();
    const num = (k) => (q.get(k) === null || q.get(k) === "" ? null : Number(q.get(k)));
    const sort = q.get("sort") || "absolute_error";
    const order = q.get("order") || "desc";
    const page = Math.max(1, Number(q.get("page") || 1));
    const pageSize = Math.min(100, Math.max(1, Number(q.get("page_size") || 15)));
    const minAtoms = num("min_atoms"), maxAtoms = num("max_atoms"), minErr = num("min_error"), maxU = num("max_uncertainty");
    if (!["id", "num_atoms", "target", "prediction", "absolute_error", "uncertainty", "error"].includes(sort)) throw new Error("bad sort key");

    let list = rows.filter((r) =>
      (minAtoms === null || r.num_atoms >= minAtoms) && (maxAtoms === null || r.num_atoms <= maxAtoms) &&
      (minErr === null || r.absolute_error >= minErr) && (maxU === null || r.uncertainty <= maxU));
    const dir = order === "asc" ? 1 : -1;
    list = list.slice().sort((a, b) => dir * (a[sort] - b[sort])); // Array.sort is stable
    const total = list.length, start = (page - 1) * pageSize;
    return { total, page, page_size: pageSize, pages: Math.max(1, Math.ceil(total / pageSize)), rows: list.slice(start, start + pageSize) };
  }

  async function molecule(id) {
    const { rows, sortedU, sortedE } = await db();
    if (!(id >= 0 && id < rows.length)) throw new Error(`Molecule id must be between 0 and ${rows.length - 1}`);
    const r = rows[id], u = r.uncertainty, ae = r.absolute_error, pred = r.prediction;
    return {
      id: r.id, num_atoms: r.num_atoms, target: r.target, prediction: pred, signed_error: r.error,
      absolute_error: ae, uncertainty: u,
      interval_1sigma: [pred - u, pred + u], interval_2sigma: [pred - 2 * u, pred + 2 * u],
      within_1sigma: ae <= u, within_2sigma: ae <= 2 * u,
      uncertainty_percentile: fracBelow(sortedU, u) * 100, error_percentile: fracBelow(sortedE, ae) * 100,
    };
  }

  window.QM9_STATIC_API = async (path) => {
    const url = new URL(path, "http://x");
    const p = url.pathname;
    if (FILES[p]) return getJson(FILES[p]);
    if (p === "/api/molecules") return browse(url.searchParams);
    const m = p.match(/^\/api\/molecules\/(\d+)$/);
    if (m) return molecule(Number(m[1]));
    throw new Error(`No static data for ${path}`);
  };
})();
