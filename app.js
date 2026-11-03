/* QM9·GraphMol Lab — frontend. All numbers come from the FastAPI backend (/api/*). */
(() => {
  const C = {
    teal: "#2dd4bf", amber: "#fbbf24", rose: "#fb7185", violet: "#a78bfa", blue: "#60a5fa",
    muted: "#8fa3b8", grid: "rgba(143,163,184,.14)",
  };
  const MODEL_COLORS = {
    "Original GCN": C.rose, "Improved GCN": C.amber, "GCN Comparison": C.teal, "GCN": C.teal,
    "GraphSAGE": C.blue, "GAT": C.violet,
  };
  Chart.defaults.color = C.muted;
  Chart.defaults.borderColor = C.grid;
  Chart.defaults.font.family = '"IBM Plex Sans", system-ui, sans-serif';
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.labels.boxWidth = 12;

  const $ = (s, el = document) => el.querySelector(s);
  const fmt = (n, d = 3) => Number(n).toFixed(d);
  const pct = (n, d = 1) => (n * 100).toFixed(d) + "%";
  const pctile = (p) => (Math.floor(p * 10) / 10).toFixed(1); // floor so 99.99 never rounds up to 100.0
  const api = async (path) => {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path} → ${r.status}`);
    return r.json();
  };
  const axis = (title, extra = {}) => ({ title: { display: !!title, text: title }, grid: { color: C.grid }, ...extra });
  const kpi = (v, l, cls = "") => `<div class="kpi ${cls}"><div class="v">${v}</div><div class="l">${l}</div></div>`;

  // Abramowitz–Stegun erf, used for the ideal Gaussian coverage curve.
  const erf = (x) => {
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  };

  /* ---------------------------------------------------------------- overview */
  async function loadOverview(models, unc) {
    const o = await api("/api/overview");
    const d = o.dataset, b = o.best_model;
    $("#kpis").innerHTML =
      kpi(d.molecules.toLocaleString(), "molecules in QM9") +
      kpi(fmt(b.r2, 3), `best test R² · ${b.name}`, "accent") +
      kpi(fmt(b.mae, 3) + " eV", "best test MAE") +
      kpi(d.split.test.toLocaleString(), "held-out test molecules");

    const first = models.find((m) => m.experiment === "Original GCN");
    const improved = models.find((m) => m.experiment === "Improved GCN");
    const s = unc.stats, bins = unc.bins;
    $("#findings").innerHTML = [
      `<div class="finding"><b>${b.name}</b> is the strongest model: R² ${fmt(b.r2)} and MAE ${fmt(b.mae)} eV on the test set.</div>`,
      `<div class="finding"><b>Training recipe matters.</b> Going from the original to the improved GCN (3 layers, 128 units, batch‑norm, dropout, scaled target) lifted R² from ${fmt(first.r2, 2)} to ${fmt(improved.r2, 2)}.</div>`,
      `<div class="finding"><b>Uncertainty is informative.</b> Error rises from ${fmt(bins[0].mae)} eV in the most‑certain fifth to ${fmt(bins[bins.length - 1].mae)} eV in the least‑certain fifth (Pearson r = ${fmt(s.pearson, 2)}).</div>`,
      `<div class="finding warn"><b>But it is overconfident.</b> ±1σ intervals cover only ${pct(s.coverage_1sigma, 0)} of molecules (a calibrated Gaussian would cover 68%).</div>`,
    ].join("");

    // dataset section
    $("#dataset-facts").innerHTML = [
      ["Molecules", d.molecules.toLocaleString()], ["Atom features", d.atom_features],
      ["Target properties", d.target_properties], ["Prediction target", `${d.target} (index ${d.target_index})`],
      ["Unit", d.unit], ["Random seed", d.seed],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    const tot = d.split.train + d.split.validation + d.split.test;
    const parts = [["Train", d.split.train, C.teal], ["Val", d.split.validation, C.amber], ["Test", d.split.test, C.blue]];
    $("#splitbar").innerHTML = parts.map(([n, v, c]) => `<div style="flex:${v};background:${c}" title="${n}: ${v.toLocaleString()}">${n} ${v.toLocaleString()}</div>`).join("");
    const t = o.target_distribution;
    $("#target-note").textContent = `Test‑set gap ranges ${fmt(t.min, 2)}–${fmt(t.max, 2)} eV (mean ${fmt(t.mean, 2)}, std ${fmt(t.std, 2)}).`;
    return o;
  }

  function targetChart(scatterPoints) {
    // histogram of the true gap from the sampled test points
    const vals = scatterPoints.map((p) => p.target);
    const lo = Math.floor(Math.min(...vals)), hi = Math.ceil(Math.max(...vals)), n = 24, w = (hi - lo) / n;
    const counts = Array(n).fill(0);
    vals.forEach((v) => counts[Math.min(n - 1, Math.floor((v - lo) / w))]++);
    new Chart($("#chart-target"), {
      type: "bar",
      data: { labels: counts.map((_, i) => (lo + i * w).toFixed(1)), datasets: [{ data: counts, backgroundColor: C.teal, borderRadius: 2 }] },
      options: { plugins: { legend: { display: false }, title: { display: true, text: "Target gap distribution (eV, 2,000‑molecule sample)", color: C.muted, font: { size: 12, weight: "400" } } },
        scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } }, y: { display: false } } },
    });
  }

  /* --------------------------------------------------------------- benchmark */
  function renderBenchmark(models) {
    const order = [...models].sort((a, b) => b.r2 - a.r2);
    const best = order[0].experiment;
    new Chart($("#chart-r2"), {
      type: "bar",
      data: { labels: order.map((m) => m.experiment), datasets: [{ data: order.map((m) => m.r2), backgroundColor: order.map((m) => MODEL_COLORS[m.experiment]), borderRadius: 6 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `R² ${fmt(c.parsed.x, 4)}` } } },
        scales: { x: { ...axis("R²"), min: 0, max: 1 }, y: { grid: { display: false } } } },
    });
    new Chart($("#chart-err"), {
      type: "bar",
      data: { labels: order.map((m) => m.experiment), datasets: [
        { label: "MAE", data: order.map((m) => m.mae), backgroundColor: C.teal, borderRadius: 4 },
        { label: "RMSE", data: order.map((m) => m.rmse), backgroundColor: C.blue, borderRadius: 4 }] },
      options: { scales: { x: { grid: { display: false } }, y: axis("eV") } },
    });
    $("#model-table tbody").innerHTML = order.map((m) =>
      `<tr class="${m.experiment === best ? "best" : ""}"><td>${m.experiment}${m.experiment === best ? '<span class="badge">best</span>' : ""}</td>
       <td>${fmt(m.mse, 4)}</td><td>${fmt(m.rmse, 4)}</td><td>${fmt(m.mae, 4)}</td><td>${fmt(m.r2, 4)}</td><td class="desc">${m.description}</td></tr>`).join("");
    const gat = models.find((m) => m.experiment === "GAT"), gcn = models.find((m) => m.experiment === "GCN Comparison"), imp = models.find((m) => m.experiment === "Improved GCN");
    $("#benchmark-note").innerHTML =
      `Each model was trained once (seed 42) for 10 epochs, so small gaps are within run‑to‑run noise: the identical 3‑layer GCN scored R² ${fmt(imp.r2, 3)} in one run and ${fmt(gcn.r2, 3)} in another. ` +
      `GAT's R² of ${fmt(gat.r2, 2)} should be read as under‑trained rather than as proof attention is worse — its training loss was still high after 10 epochs (see Training).`;
  }

  /* ---------------------------------------------------------------- training */
  async function renderHistory() {
    const h = await api("/api/training-history");
    let metric = "val_mse";
    const names = { val_mse: "Validation MSE", train_mse: "Training MSE", val_mae: "Validation MAE" };
    const build = () => Object.entries(h).map(([name, rows]) => ({
      label: name, data: rows.map((r) => r[metric]), borderColor: MODEL_COLORS[name], backgroundColor: MODEL_COLORS[name],
      tension: 0.25, pointRadius: 3, borderWidth: 2,
    }));
    const chart = new Chart($("#chart-history"), {
      type: "line",
      data: { labels: h.GCN.map((r) => r.epoch), datasets: build() },
      options: { interaction: { mode: "index", intersect: false },
        scales: { x: axis("Epoch"), y: axis(names[metric] + " (standardised)") } },
    });
    document.querySelectorAll("#history-metric button").forEach((btn) => btn.addEventListener("click", () => {
      document.querySelectorAll("#history-metric button").forEach((b) => b.classList.remove("on"));
      btn.classList.add("on"); metric = btn.dataset.m;
      chart.data.datasets = build(); chart.options.scales.y.title.text = names[metric] + " (standardised)"; chart.update();
    }));
  }

  /* ------------------------------------------------------------- uncertainty */
  async function renderUncertainty(unc) {
    const s = unc.stats, m = unc.mc_mean_model;
    $("#u-kpis").innerHTML =
      kpi(fmt(m.mae, 3) + " eV", "MC‑mean MAE (50 samples)") +
      kpi(fmt(s.pearson, 2), "Pearson r: uncertainty vs error", "accent") +
      kpi(fmt(s.low_uncertainty_mae, 3), "MAE · lower‑half uncertainty") +
      kpi(fmt(s.high_uncertainty_mae, 3), "MAE · upper‑half uncertainty") +
      kpi(pct(s.coverage_1sigma, 0), "within ±1σ (ideal ≈ 68%)") +
      kpi(pct(s.coverage_2sigma, 0), "within ±2σ (ideal ≈ 95%)");

    const { points } = await api("/api/uncertainty/scatter?limit=2000");
    targetChart(points);
    new Chart($("#chart-u-scatter"), {
      type: "scatter",
      data: { datasets: [
        { label: "Molecules", data: points.map((p) => ({ x: p.uncertainty, y: p.absolute_error })), backgroundColor: "rgba(45,212,191,.35)", pointRadius: 2.2 },
        { type: "line", label: "Quintile mean", data: unc.bins.map((b) => ({ x: b.mean_uncertainty, y: b.mae })), borderColor: C.amber, backgroundColor: C.amber, borderWidth: 3, pointRadius: 5, tension: 0.2 },
      ] },
      options: { scales: { x: axis("Predicted uncertainty σ (eV)"), y: axis("Absolute error (eV)") } },
    });
    new Chart($("#chart-u-bins"), {
      type: "bar",
      data: { labels: unc.bins.map((b) => `Q${b.bin} · σ≈${fmt(b.mean_uncertainty, 2)}`), datasets: [{ data: unc.bins.map((b) => b.mae), backgroundColor: [C.teal, "#4fc7c0", C.blue, C.violet, C.rose], borderRadius: 6 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } }, y: { ...axis("Mean absolute error (eV)"), beginAtZero: true } } },
    });
    new Chart($("#chart-retention"), {
      type: "line",
      data: { labels: unc.retention.map((r) => Math.round(r.keep * 100) + "%"), datasets: [{ label: "MAE", data: unc.retention.map((r) => r.mae), borderColor: C.teal, backgroundColor: "rgba(45,212,191,.15)", fill: true, tension: 0.25, pointRadius: 3 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: axis("Share of molecules kept (most confident first)"), y: { ...axis("MAE (eV)"), beginAtZero: true } } },
    });
    new Chart($("#chart-calibration"), {
      type: "line",
      data: { labels: unc.calibration.map((c) => c.k + "σ"), datasets: [
        { label: "Observed", data: unc.calibration.map((c) => c.observed * 100), borderColor: C.rose, backgroundColor: C.rose, tension: 0.25, pointRadius: 4, borderWidth: 3 },
        { label: "Ideal Gaussian", data: unc.calibration.map((c) => erf(c.k / Math.SQRT2) * 100), borderColor: C.muted, borderDash: [6, 5], pointRadius: 0, borderWidth: 2 },
      ] },
      options: { scales: { x: axis("Interval half‑width k·σ"), y: { ...axis("Molecules inside interval (%)"), min: 0, max: 100 } } },
    });
    const half = unc.retention.find((r) => r.keep === 0.5);
    $("#u-note").innerHTML =
      `Takeaway: σ ranks molecules by difficulty reasonably well — keeping the most‑confident half drops MAE from ${fmt(m.mae)} to ${fmt(half.mae)} eV — ` +
      `but the raw σ values are too small to use as literal error bars (the Spearman rank correlation is only ${fmt(s.spearman, 2)}). ` +
      `Recalibrating σ (for example by scaling it on the validation set) or using deep ensembles would be natural next steps.`;
  }

  /* ------------------------------------------------------------------ errors */
  async function renderErrors() {
    const e = await api("/api/errors");
    const sizes = e.by_size;
    new Chart($("#chart-size"), {
      data: { labels: sizes.map((s) => s.num_atoms), datasets: [
        { type: "bar", label: "MAE (eV)", data: sizes.map((s) => s.mae), backgroundColor: sizes.map((s) => (s.count < 30 ? "rgba(251,191,36,.45)" : C.teal)), borderRadius: 3, yAxisID: "y" },
        { type: "line", label: "Molecules", data: sizes.map((s) => s.count), borderColor: C.violet, pointRadius: 0, borderWidth: 2, yAxisID: "y1" },
      ] },
      options: { plugins: { tooltip: { callbacks: { afterBody: (it) => (sizes[it[0].dataIndex].count < 30 ? "⚠ fewer than 30 molecules" : "") } } },
        scales: { x: axis("Atoms in molecule", { grid: { display: false } }), y: { ...axis("MAE (eV)"), beginAtZero: true },
          y1: { position: "right", grid: { display: false }, title: { display: true, text: "Molecules" }, beginAtZero: true } } },
    });
    document.querySelector("#chart-size").closest(".card").insertAdjacentHTML("beforeend", '<p class="muted small">Amber bars have fewer than 30 test molecules, so treat them as noisy.</p>');
    const h = e.histogram, mids = h.counts.map((_, i) => (h.edges[i] + h.edges[i + 1]) / 2);
    new Chart($("#chart-hist"), {
      type: "bar",
      data: { labels: mids.map((v) => v.toFixed(2)), datasets: [{ data: h.counts, backgroundColor: mids.map((v) => (Math.abs(v) < 0.25 ? C.teal : C.blue)), borderRadius: 2 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: { ...axis("Prediction − target (eV)", { grid: { display: false } }), ticks: { maxTicksLimit: 9 } }, y: axis("Molecules") } },
    });
    document.querySelector("#chart-hist").closest(".card").insertAdjacentHTML("beforeend",
      `<p class="muted small">Mean signed error ${fmt(e.mc_mean_model.bias, 3)} eV; ${h.outside_range} molecules fall outside ±2 eV.</p>`);

    const { molecules } = await api("/api/difficult");
    $("#difficult-table tbody").innerHTML = molecules.map((r, i) =>
      `<tr><td>${i + 1}</td><td>${r.num_atoms}</td><td>${fmt(r.target, 2)}</td><td>${fmt(r.prediction, 2)}</td>
       <td>${r.error > 0 ? "+" : ""}${fmt(r.error, 2)}</td><td>${fmt(r.uncertainty, 3)}</td><td>${fmt(r.difficulty_score, 3)}</td></tr>`).join("");
  }

  /* ---------------------------------------------------------------- explorer */
  const st = { page: 1, selected: null, pages: 1 };
  const val = (id) => $(id).value.trim();

  async function loadTable() {
    const q = new URLSearchParams({ sort: val("#f-sort"), order: val("#f-order"), page: st.page, page_size: 12 });
    if (val("#f-min")) q.set("min_atoms", val("#f-min"));
    if (val("#f-max")) q.set("max_atoms", val("#f-max"));
    if (val("#f-maxu")) q.set("max_uncertainty", val("#f-maxu"));
    const r = await api("/api/molecules?" + q);
    st.pages = r.pages;
    $("#mol-table tbody").innerHTML = r.rows.map((m) =>
      `<tr class="clickable ${m.id === st.selected ? "sel" : ""}" data-id="${m.id}"><td>${m.id}</td><td>${m.num_atoms}</td><td>${fmt(m.target, 2)}</td>
       <td>${fmt(m.prediction, 2)}</td><td>${fmt(m.absolute_error, 3)}</td><td>${fmt(m.uncertainty, 3)}</td></tr>`).join("") ||
      '<tr><td colspan="6" class="muted">No molecules match these filters.</td></tr>';
    $("#p-info").textContent = `Page ${r.page} of ${r.pages} · ${r.total.toLocaleString()} molecules`;
    $("#p-prev").disabled = r.page <= 1; $("#p-next").disabled = r.page >= r.pages;
    document.querySelectorAll("#mol-table tr.clickable").forEach((tr) => tr.addEventListener("click", () => showMolecule(+tr.dataset.id)));
  }

  async function showMolecule(id) {
    st.selected = id;
    document.querySelectorAll("#mol-table tr.clickable").forEach((tr) => tr.classList.toggle("sel", +tr.dataset.id === id));
    const d = $("#detail");
    try {
      const m = await api("/api/molecules/" + id);
      const lo = Math.min(m.target, m.interval_2sigma[0]), hi = Math.max(m.target, m.interval_2sigma[1]);
      const pad = (hi - lo) * 0.12 || 0.1, a = lo - pad, b = hi + pad;
      const x = (v) => ((v - a) / (b - a)) * 100 + "%";
      const w = (l, h) => ((h - l) / (b - a)) * 100 + "%";
      d.innerHTML = `
        <h3>Test molecule #${m.id} <span class="badge ${m.within_2sigma ? "" : "bad"}">${m.within_1sigma ? "inside ±1σ" : m.within_2sigma ? "inside ±2σ" : "outside ±2σ"}</span></h3>
        <p class="muted small">${m.num_atoms} atoms · HOMO‑LUMO gap in eV</p>
        <div class="big">
          <div><b>${fmt(m.target, 3)}</b><span>actual gap</span></div>
          <div><b style="color:var(--teal)">${fmt(m.prediction, 3)}</b><span>predicted (MC mean)</span></div>
          <div><b>${fmt(m.absolute_error, 3)}</b><span>absolute error · larger than ${pctile(m.error_percentile)}% of test errors</span></div>
          <div><b>±${fmt(m.uncertainty, 3)}</b><span>σ · larger than ${pctile(m.uncertainty_percentile)}% of test σ values</span></div>
        </div>
        <div class="interval">
          <div class="axis"></div>
          <div class="band2" style="left:${x(m.interval_2sigma[0])};width:${w(m.interval_2sigma[0], m.interval_2sigma[1])}"></div>
          <div class="band1" style="left:${x(m.interval_1sigma[0])};width:${w(m.interval_1sigma[0], m.interval_1sigma[1])}"></div>
          <div class="pt pred" style="left:${x(m.prediction)}"></div><div class="lbl pred" style="left:${x(m.prediction)}">pred</div>
          <div class="pt true" style="left:${x(m.target)}"></div><div class="lbl true" style="left:${x(m.target)}">actual</div>
        </div>
        <div class="legend"><span><i style="background:rgba(45,212,191,.6)"></i>±1σ</span><span><i style="background:rgba(96,165,250,.4)"></i>±2σ</span></div>`;
    } catch (err) { d.innerHTML = `<p class="error">${err.message}</p>`; }
  }

  function initExplorer() {
    const reload = () => { st.page = 1; loadTable(); };
    $("#f-apply").addEventListener("click", () => {
      const id = val("#f-id");
      if (id !== "") showMolecule(Math.max(0, Math.min(13083, +id)));
      reload();
    });
    ["#f-sort", "#f-order"].forEach((s) => $(s).addEventListener("change", reload));
    $("#p-prev").addEventListener("click", () => { st.page = Math.max(1, st.page - 1); loadTable(); });
    $("#p-next").addEventListener("click", () => { st.page = Math.min(st.pages, st.page + 1); loadTable(); });
    return loadTable().then(() => { const first = $("#mol-table tr.clickable"); if (first) showMolecule(+first.dataset.id); });
  }

  /* --------------------------------------------------------------- nav + boot */
  function initNav() {
    const links = [...document.querySelectorAll(".nav nav a")];
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
    }), { rootMargin: "-40% 0px -55% 0px" });
    document.querySelectorAll("main section").forEach((s) => io.observe(s));
  }

  async function boot() {
    initNav();
    try {
      const [{ models }, unc] = await Promise.all([api("/api/models"), api("/api/uncertainty")]);
      await loadOverview(models, unc);
      renderBenchmark(models);
      await Promise.all([renderHistory(), renderUncertainty(unc), renderErrors(), initExplorer()]);
      if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
    } catch (err) {
      console.error(err);
      $("#kpis").innerHTML = `<p class="error">Could not load data from the API: ${err.message}. Is the backend running?</p>`;
    }
  }
  boot();
})();
