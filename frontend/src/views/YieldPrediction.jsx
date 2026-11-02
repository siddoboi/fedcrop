import { useMemo, useState } from "react";
import {
  CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { useApi } from "../api.js";
import { ARM_COLORS, armLabel, fmt } from "../labels.js";
import { C, axisProps, gridProps, tooltipProps } from "../components/charts.jsx";
import {
  Callout, Card, Controls, PageHeader, Provenance, Section, Select, Status,
} from "../components/ui.jsx";

const ALL = "__all__";

export default function YieldPrediction() {
  const { data, error, loading } = useApi("/api/predictions", "/api/baselines");
  const [state, setState] = useState(ALL);
  const [arm, setArm] = useState("fedper");

  const view = useMemo(() => {
    if (!data) return null;
    const [pred] = data;
    const rows = state === ALL ? pred.rows : pred.rows.filter((r) => r.client === state);
    const points = rows
      .filter((r) => r.actual != null && r[`pred_${arm}`] != null)
      .map((r) => ({ actual: r.actual, pred: r[`pred_${arm}`], client: r.client }));
    const vals = points.flatMap((p) => [p.actual, p.pred]);
    const lo = Math.floor(Math.min(...vals) / 500) * 500;
    const hi = Math.ceil(Math.max(...vals) / 500) * 500;
    return { points, lo, hi };
  }, [data, state, arm]);

  if (!data) return <Status loading={loading} error={error} />;
  const [pred, base] = data;
  const arms = pred.arms;

  const metrics = state === ALL
    ? pred.overall
    : pred.per_client.find((p) => p.client === state)?.metrics || {};
  const trend = state === ALL
    ? base.table.find((t) => t.model === "district_trend")
    : base.per_client_trend.find((t) => t.client === state);

  return (
    <>
      <PageHeader kicker="Crop yield prediction" title="How close the predictions come">
        Every point is one district in one test year (2012–2015), which the models never
        saw in training. A perfect model puts every point on the diagonal.
      </PageHeader>

      <Controls>
        <Select id="yp-state" label="State" value={state} onChange={setState}
          options={[{ value: ALL, label: `All ${pred.clients.length} states` },
            ...pred.clients.map((c) => ({ value: c, label: c }))]} />
        <Select id="yp-arm" label="Method" value={arm} onChange={setArm}
          options={arms.map((a) => ({ value: a, label: armLabel(a) }))} />
      </Controls>

      <div className="grid-2-wide" style={{ marginBottom: 40 }}>
        <Card title={`${armLabel(arm)} · ${state === ALL ? "all states" : state} · ${view.points.length} district-years`}>
          <ResponsiveContainer width="100%" height={380}>
            <ScatterChart margin={{ top: 10, right: 18, bottom: 26, left: 8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis type="number" dataKey="actual" name="Actual" domain={[view.lo, view.hi]} {...axisProps}
                tickFormatter={fmt.i}
                label={{ value: "actual yield (kg/ha)", position: "insideBottom", offset: -14, fill: C.muted, fontSize: 12 }} />
              <YAxis type="number" dataKey="pred" name="Predicted" domain={[view.lo, view.hi]} {...axisProps}
                tickFormatter={fmt.i} width={56}
                label={{ value: "predicted (kg/ha)", angle: -90, position: "insideLeft", fill: C.muted, fontSize: 12, dy: 50 }} />
              <ReferenceLine segment={[{ x: view.lo, y: view.lo }, { x: view.hi, y: view.hi }]}
                stroke={C.muted} strokeDasharray="4 4" ifOverflow="hidden" />
              <Tooltip {...tooltipProps} cursor={{ strokeDasharray: "3 3", stroke: C.line }}
                formatter={(v, n) => [`${fmt.i(v)} kg/ha`, n]} />
              <Scatter data={view.points} fill={ARM_COLORS[arm]} fillOpacity={0.55} isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="chart-note">Points above the diagonal are over-predictions; below, under-predictions. Single seed ({pred.seed}).</p>
        </Card>

        <Card title={state === ALL ? "Error, all states" : `Error, ${state}`}>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Method</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead>
              <tbody>
                {arms.map((a) => {
                  const m = metrics[a];
                  if (!m) return null;
                  return (
                    <tr key={a} className={a === arm ? "hl" : ""}>
                      <td className="name">{armLabel(a)}</td>
                      <td>{fmt.i(m.rmse)}</td>
                      <td>{fmt.i(m.mae)}</td>
                      <td className={m.r2 != null && m.r2 < 0 ? "neg" : ""}>{fmt.n(m.r2)}</td>
                    </tr>
                  );
                })}
                {trend && (
                  <tr className="baseline">
                    <td className="name">District trend</td>
                    <td>{fmt.i(trend.rmse)}</td>
                    <td>{fmt.i(trend.mae)}</td>
                    <td className={trend.r2 < 0 ? "neg" : ""}>{fmt.n(trend.r2)}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="chart-note">RMSE and MAE in kg/ha. The trend baseline extends each district's own 1990–2009 trend.</p>
          {state !== ALL && metrics[arm]?.r2 != null && metrics[arm].r2 < 0 && (
            <Callout warn title="Negative R².">
              Within this state the model does worse than predicting the state's own average
              yield every year. Pooled across states the same model can still look good, because
              differences between states are easy to capture.
            </Callout>
          )}
        </Card>
      </div>

      <Section title="Error by state" tag="Heterogeneity"
        lede="RMSE in kg/ha for each method in each state. The lowest error in each row is highlighted. Select a row to inspect that state.">
        <Card>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>State</th><th>District-years</th>
                  {arms.map((a) => <th key={a}>{armLabel(a)}</th>)}
                  <th>Trend</th>
                </tr>
              </thead>
              <tbody>
                {pred.per_client.map((p) => {
                  const best = arms.reduce((b, a) =>
                    (p.metrics[a]?.rmse ?? Infinity) < (p.metrics[b]?.rmse ?? Infinity) ? a : b, arms[0]);
                  const t = base.per_client_trend.find((x) => x.client === p.client);
                  return (
                    <tr key={p.client} className={p.client === state ? "hl" : ""}
                      style={{ cursor: "pointer" }} onClick={() => setState(p.client)}>
                      <td className="name">{p.client}</td>
                      <td className="sd">{p.n}</td>
                      {arms.map((a) => (
                        <td key={a} className={a === best ? "pos" : ""}>{fmt.i(p.metrics[a]?.rmse)}</td>
                      ))}
                      <td className="muted">{fmt.i(t?.rmse)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </Section>

      <Provenance />
    </>
  );
}
