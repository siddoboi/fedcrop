import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { apiUrl, useApi } from "../api.js";
import { ARM_COLORS, armLabel, featureKind, fmt, pretty } from "../labels.js";
import { C, axisProps, gridProps, legendProps, tooltipProps } from "../components/charts.jsx";
import {
  Callout, Card, KV, PageHeader, Provenance, Section, Select, Stat, Status,
} from "../components/ui.jsx";

const PAGE = 25;

/** Quote a CSV field when it needs it. */
const cell = (v) => {
  if (v == null) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

function downloadCsv(name, header, rows) {
  const text = [header, ...rows].map((r) => r.map(cell).join(",")).join("\n") + "\n";
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function postForm(path, form) {
  const res = await fetch(apiUrl(path), { method: "POST", body: form });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}

export default function Predict() {
  const { data, error, loading } = useApi("/api/upload/status", "/api/upload/columns");
  const [file, setFile] = useState(null);
  const [primary, setPrimary] = useState("fedper");
  const [compare, setCompare] = useState([]);
  const [busy, setBusy] = useState(false);
  const [fail, setFail] = useState(null);
  const [result, setResult] = useState(null);
  const [drag, setDrag] = useState(false);
  const input = useRef(null);
  const resultsRef = useRef(null);

  if (!data) return <Status loading={loading} error={error} />;
  const [status, cols] = data;
  const arms = status.arms_available;
  const others = arms.filter((a) => a !== primary);

  async function run(f) {
    if (!f) return;
    setBusy(true); setFail(null); setResult(null);
    const form = new FormData();
    form.append("file", f);
    form.append("models", [primary, ...compare.filter((c) => c !== primary)].join(","));
    try {
      const body = await postForm("/api/upload/predict", form);
      setResult({ ...body, fileName: f.name });
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch (e) {
      setFail(e.message);
    } finally {
      setBusy(false);
    }
  }

  const toggle = (a) => setCompare((c) => (c.includes(a) ? c.filter((x) => x !== a) : [...c, a]));

  return (
    <>
      <PageHeader kicker="Predict from CSV" title="Run the trained models on your own rows">
        Upload district-years in the project's format. Each row is predicted with its own
        state's model and explained against that state's typical year. Nothing you upload is
        stored.
      </PageHeader>

      {!status.ready && (
        <Callout warn title="Predictions are not available yet.">
          {status.problems.join(" ")}
        </Callout>
      )}

      <Section title="Get a file" tag="Format"
        lede={`A row is one district in one year: 60 monthly climate values (12 months × 5 variables), 6 annual farm figures, and optionally the actual yield. The three farm ratios are computed for you. Only the ${status.states.length} federated states are accepted.`}>
        <Card>
          <p className="lede" style={{ fontSize: 13.5 }}>
            Start from the blank template so the 71 column names are exact. Column order does not
            matter; names do. Ready-made examples from the test years are in the
            repo's <code>demo_data</code> folder.
          </p>
          <a className="btn" href={apiUrl("/api/upload/template")} download>Download blank template</a>
          <details className="cols">
            <summary>Column guide</summary>
            <div className="table-wrap" style={{ maxHeight: 260, overflowY: "auto", marginTop: 10 }}>
              <table>
                <thead><tr><th>Column</th><th>Unit</th><th>Required</th></tr></thead>
                <tbody>
                  {cols.columns.map((c) => (
                    <tr key={c.name}>
                      <td className="name" style={{ fontFamily: "ui-monospace, Consolas, monospace", fontSize: 12.5 }}>{c.name}</td>
                      <td>{c.unit}</td>
                      <td className={c.required ? "" : "muted"}>{c.required ? "yes" : "optional"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="chart-note">Computed by the server: {cols.derived_by_server.map(pretty).join(", ")}.</p>
          </details>
        </Card>
      </Section>

      <Section title="Upload and predict" tag="Models">
        <Card>
          <div className="upload-grid">
            <div
              className={`drop${drag ? " over" : ""}`}
              onClick={() => input.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; if (f) setFile(f); }}
              role="button" tabIndex={0}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") input.current?.click(); }}>
              <input ref={input} type="file" accept=".csv,text/csv" hidden id="csv-file"
                onChange={(e) => setFile(e.target.files?.[0] || null)} />
              {file ? (
                <><strong>{file.name}</strong><span>{(file.size / 1024).toFixed(1)} KB · select to change</span></>
              ) : (
                <><strong>Drop a CSV here</strong><span>or select to choose a file · up to {status.max_rows} rows, {status.max_mb} MB</span></>
              )}
            </div>
            <div className="stack" style={{ gap: 14 }}>
              <Select id="pr-primary" label="Primary model" value={primary}
                onChange={(v) => { setPrimary(v); setCompare((c) => c.filter((x) => x !== v)); }}
                options={arms.map((a) => ({ value: a, label: armLabel(a) + (a === "fedper" ? " (recommended)" : "") }))} />
              <div className="checks" role="group" aria-label="Also compare with">
                <span>Also compare with</span>
                {others.map((a) => (
                  <label key={a}>
                    <input type="checkbox" checked={compare.includes(a)} onChange={() => toggle(a)} />
                    {armLabel(a)}
                  </label>
                ))}
              </div>
              <button type="button" className="btn primary" disabled={!file || busy || !status.ready}
                onClick={() => run(file)}>
                {busy ? "Predicting…" : "Predict"}
              </button>
            </div>
          </div>
          {fail && <Callout warn title="The file could not be processed.">{fail}</Callout>}
        </Card>
      </Section>

      <div ref={resultsRef}>
        {result && <Results r={result} />}
      </div>

      <Provenance />
    </>
  );
}

function Results({ r }) {
  const s = r.summary;
  const arms = [s.primary, ...s.compared];
  const [page, setPage] = useState(0);
  const [sel, setSel] = useState(r.rows[0]?.row ?? null);
  useEffect(() => { setPage(0); setSel(r.rows[0]?.row ?? null); }, [r]);

  const pm = r.metrics?.[s.primary];
  const pages = Math.max(1, Math.ceil(r.rows.length / PAGE));
  const shown = r.rows.slice(page * PAGE, page * PAGE + PAGE);
  const selected = r.rows.find((x) => x.row === sel);

  const byState = useMemo(() => {
    const g = {};
    r.rows.forEach((x) => { (g[x.state] ||= []).push(x); });
    return Object.entries(g).map(([state, list]) => {
      const o = { state, n: list.length };
      arms.forEach((a) => {
        const v = list.map((x) => x.pred[a]).filter((v) => v != null);
        o[a] = v.length ? v.reduce((p, q) => p + q, 0) / v.length : null;
      });
      const act = list.map((x) => x.actual).filter((v) => v != null);
      o.actual = act.length ? act.reduce((p, q) => p + q, 0) / act.length : null;
      return o;
    });
  }, [r]); // eslint-disable-line react-hooks/exhaustive-deps

  const scatter = r.rows.filter((x) => x.actual != null && x.pred[s.primary] != null)
    .map((x) => ({ actual: x.actual, pred: x.pred[s.primary], label: `${x.state} ${x.district_name || x.district_code} ${x.year ?? ""}` }));
  const vals = scatter.flatMap((p) => [p.actual, p.pred]);
  const lo = vals.length ? Math.floor(Math.min(...vals) / 500) * 500 : 0;
  const hi = vals.length ? Math.ceil(Math.max(...vals) / 500) * 500 : 1;

  function exportResults() {
    const header = ["line", "state", "district_code", "district_name", "year", ...r.input_order,
      "actual_yield", ...arms.map((a) => `pred_${a}`),
      ...(s.has_actual ? [`error_${s.primary}`] : []), "warnings"];
    const rows = r.rows.map((x) => [x.row, x.state, x.district_code, x.district_name, x.year, ...x.x,
      x.actual, ...arms.map((a) => (x.pred[a] == null ? "" : x.pred[a].toFixed(1))),
      ...(s.has_actual ? [x.actual != null && x.pred[s.primary] != null ? (x.pred[s.primary] - x.actual).toFixed(1) : ""] : []),
      x.warnings.join(" | ")]);
    downloadCsv(r.fileName.replace(/\.csv$/i, "") + "_predictions.csv", header, rows);
  }
  function exportRejected() {
    downloadCsv(r.fileName.replace(/\.csv$/i, "") + "_rejected.csv", ["line", "state", "reasons"],
      r.rejected.map((x) => [x.row, x.state, x.reasons.join(" ")]));
  }

  return (
    <Section title={`Results · ${r.fileName}`} tag={`seed ${s.seed}`}
      lede={`Predicted with ${armLabel(s.primary)}${s.compared.length ? `, compared with ${s.compared.map(armLabel).join(", ")}` : ""}. Yields in kg/ha.`}>
      <div className="stats" style={{ marginTop: 0, marginBottom: 20 }}>
        <Stat value={s.accepted} label="rows predicted" note={`of ${s.rows_total} in the file`} />
        <Stat value={s.rejected} label="rows rejected" tone={s.rejected ? "neg" : ""} />
        <Stat value={s.states.length} label="states" note={s.states.join(", ")} />
        <Stat value={s.rows_with_warnings} label="rows with warnings" />
        {pm && <Stat value={fmt.i(pm.rmse)} label={`RMSE, ${armLabel(s.primary)}`} note="kg/ha" />}
        {pm && <Stat value={fmt.n(pm.r2)} label={`R², ${armLabel(s.primary)}`} tone={pm.r2 != null && pm.r2 < 0 ? "neg" : ""} />}
      </div>

      <div className="actions">
        <button type="button" className="btn primary" onClick={exportResults} disabled={!r.rows.length}>
          Download results CSV
        </button>
        {r.rejected.length > 0 && <button type="button" className="btn ghost" onClick={exportRejected}>Download rejected rows</button>}
      </div>

      {r.notices.map((n) => <Callout key={n} warn>{n}</Callout>)}

      {r.rejected.length > 0 && (
        <Card title={`Rejected rows (${r.rejected.length})`} className="mt">
          <div className="table-wrap" style={{ maxHeight: 280, overflowY: "auto" }}>
            <table>
              <thead><tr><th>Line</th><th style={{ textAlign: "left" }}>State</th><th style={{ textAlign: "left" }}>Reason</th></tr></thead>
              <tbody>
                {r.rejected.map((x) => (
                  <tr key={x.row}>
                    <td className="name">{x.row}</td>
                    <td style={{ textAlign: "left" }}>{x.state || "—"}</td>
                    <td className="wrap">{x.reasons.join(" ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {r.rows.length > 0 && (
        <div className={`${s.has_actual ? "grid-2" : ""} mt`}>
          {s.has_actual && (
            <Card title={`Predicted vs actual · ${armLabel(s.primary)}`}>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 10, right: 16, bottom: 26, left: 8 }}>
                  <CartesianGrid {...gridProps} />
                  <XAxis type="number" dataKey="actual" name="Actual" domain={[lo, hi]} {...axisProps} tickFormatter={fmt.i}
                    label={{ value: "actual (kg/ha)", position: "insideBottom", offset: -14, fill: C.muted, fontSize: 12 }} />
                  <YAxis type="number" dataKey="pred" name="Predicted" domain={[lo, hi]} {...axisProps} tickFormatter={fmt.i} width={56} />
                  <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke={C.muted} strokeDasharray="4 4" ifOverflow="hidden" />
                  <Tooltip {...tooltipProps} cursor={{ strokeDasharray: "3 3", stroke: C.line }}
                    formatter={(v, n) => [`${fmt.i(v)} kg/ha`, n]}
                    labelFormatter={() => ""} />
                  <Scatter data={scatter} fill={ARM_COLORS[s.primary]} fillOpacity={0.7} isAnimationActive={false} />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="chart-note">Dashed line: perfect prediction. Above it, over-prediction.</p>
            </Card>
          )}
          <Card title="Average predicted yield by state">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={byState} margin={{ top: 4, right: 10, bottom: 4, left: 4 }}>
                <CartesianGrid {...gridProps} vertical={false} />
                <XAxis dataKey="state" {...axisProps} interval={0} tick={{ fill: C.muted, fontSize: 11 }} />
                <YAxis {...axisProps} tickFormatter={fmt.i} width={52} />
                <Tooltip {...tooltipProps} formatter={(v, n) => [`${fmt.i(v)} kg/ha`, n === "actual" ? "Actual" : armLabel(n)]} />
                <Legend {...legendProps} verticalAlign="top" height={28} formatter={(n) => (n === "actual" ? "Actual" : armLabel(n))} />
                {s.has_actual && <Bar dataKey="actual" fill={C.muted} isAnimationActive={false} />}
                {arms.map((a) => <Bar key={a} dataKey={a} fill={ARM_COLORS[a]} isAnimationActive={false} />)}
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {r.metrics && arms.length > 1 && (
        <Card title="Accuracy by model" className="mt">
          <div className="table-wrap">
            <table>
              <thead><tr><th>Model</th><th>Rows</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead>
              <tbody>
                {arms.map((a) => {
                  const m = r.metrics[a];
                  return (
                    <tr key={a} className={a === s.primary ? "hl" : ""}>
                      <td className="name">{armLabel(a)}</td>
                      <td>{m.n}</td><td>{fmt.i(m.rmse)}</td><td>{fmt.i(m.mae)}</td>
                      <td className={m.r2 != null && m.r2 < 0 ? "neg" : ""}>{fmt.n(m.r2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {r.rows.length > 0 && (
        <div className="grid-2-wide mt">
          <Card title="Rows · select one to explain it">
            <div className="table-wrap">
              <table className="rows">
                <thead>
                  <tr>
                    <th>Line</th><th style={{ textAlign: "left" }}>State</th><th style={{ textAlign: "left" }}>District</th><th>Year</th>
                    {s.has_actual && <th>Actual</th>}
                    {arms.map((a) => <th key={a}>{armLabel(a)}</th>)}
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((x) => (
                    <tr key={x.row} className={x.row === sel ? "hl" : ""} onClick={() => setSel(x.row)}
                      tabIndex={0} onKeyDown={(e) => { if (e.key === "Enter") setSel(x.row); }}>
                      <td className="name">{x.row}</td>
                      <td style={{ textAlign: "left" }}>{x.state}</td>
                      <td style={{ textAlign: "left" }}>{x.district_name || x.district_code || "—"}</td>
                      <td>{x.year ?? "—"}</td>
                      {s.has_actual && <td className="muted">{fmt.i(x.actual)}</td>}
                      {arms.map((a) => <td key={a}>{fmt.i(x.pred[a])}</td>)}
                      <td title={x.warnings.join("\n")}>{x.warnings.length ? <span className="warn-dot">!</span> : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {pages > 1 && (
              <div className="pager">
                <button type="button" className="btn ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
                <span>Page {page + 1} of {pages}</span>
                <button type="button" className="btn ghost" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next</button>
              </div>
            )}
          </Card>
          {selected && <Explain row={selected} arms={arms.filter((a) => selected.pred[a] != null)} primary={s.primary} />}
        </div>
      )}
    </Section>
  );
}

function Explain({ row, arms, primary }) {
  const [arm, setArm] = useState(arms.includes(primary) ? primary : arms[0]);
  const [state, setState] = useState({ data: null, error: null, loading: true });

  useEffect(() => { if (!arms.includes(arm)) setArm(arms[0]); }, [row]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!arm) return;
    let live = true;
    setState({ data: null, error: null, loading: true });
    fetch(apiUrl("/api/upload/explain"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: row.state, arm, x: row.x }),
    })
      .then(async (res) => {
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.detail || res.statusText);
        return body;
      })
      .then((d) => live && setState({ data: d, error: null, loading: false }))
      .catch((e) => live && setState({ data: null, error: e, loading: false }));
    return () => { live = false; };
  }, [row, arm]);

  const d = state.data;
  const top = d ? d.contributions.slice(0, 12).map((c) => ({
    name: pretty(c.feature), value: c.contribution, kind: featureKind(c.feature),
  })) : [];

  return (
    <Card title={`Why line ${row.row}: ${row.district_name || row.district_code || row.state}, ${row.year ?? ""}`}>
      {arms.length > 1 && (
        <div style={{ marginBottom: 12 }}>
          <Select id="ex-arm" label="Explain model" value={arm} onChange={setArm}
            options={arms.map((a) => ({ value: a, label: armLabel(a) }))} />
        </div>
      )}
      {state.error ? <Callout warn title="Could not explain this row.">{state.error.message}</Callout>
        : !d ? <p className="muted">Computing the explanation…</p> : (
          <>
            <KV k={`Typical year in ${d.state}`} v={`${fmt.i(d.typical_year_prediction)} kg/ha`} />
            <KV k="This row" v={`${fmt.i(d.prediction)} kg/ha`} />
            <KV k="Difference to explain" v={`${d.actual_gap >= 0 ? "+" : "−"}${fmt.i(Math.abs(d.actual_gap))} kg/ha`}
              tone={d.actual_gap >= 0 ? "pos" : "neg"} />
            <ResponsiveContainer width="100%" height={top.length * 26 + 30}>
              <BarChart data={top} layout="vertical" margin={{ top: 12, right: 44, bottom: 0, left: 4 }}>
                <CartesianGrid {...gridProps} horizontal={false} />
                <XAxis type="number" {...axisProps} tickFormatter={(v) => fmt.i(v)} />
                <YAxis type="category" dataKey="name" {...axisProps} width={190} interval={0} tick={{ fill: C.text, fontSize: 12 }} />
                <ReferenceLine x={0} stroke={C.muted} />
                <Tooltip {...tooltipProps} formatter={(v) => [`${v >= 0 ? "+" : "−"}${fmt.i(Math.abs(v))} kg/ha`, "Contribution"]} />
                <Bar dataKey="value" isAnimationActive={false}>
                  {top.map((t) => <Cell key={t.name} fill={t.value >= 0 ? C.moss : C.clay} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="chart-note">
              GradientSHAP, in kg/ha, compared with {d.baseline}. Green raised this prediction, red lowered it.
              Top 12 of 69 inputs; together all 69 account for {d.explained_gap >= 0 ? "+" : "−"}{fmt.i(Math.abs(d.explained_gap))} kg/ha of the difference.
            </p>
          </>
        )}
    </Card>
  );
}
