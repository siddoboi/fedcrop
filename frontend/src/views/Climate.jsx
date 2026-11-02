import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useApi } from "../api.js";
import { armLabel, fmt } from "../labels.js";
import { C, axisProps, divergingFill, gridProps, tooltipProps } from "../components/charts.jsx";
import {
  Callout, Card, Controls, KV, PageHeader, Provenance, Section, Segmented, Select, Status,
} from "../components/ui.jsx";

const FED = "__federation__";
const TEMPS = [0, 1, 2];
const RAINS = [-20, 0, 20];
const tLabel = (t) => (t === 0 ? "Observed" : `+${t} °C`);
const rLabel = (r) => (r === 0 ? "Observed" : `${r > 0 ? "+" : "−"}${Math.abs(r)}%`);

export default function Climate() {
  const { data, error, loading } = useApi("/api/scenarios", "/api/ood", "/api/confound?limit=10");
  const [state, setState] = useState(FED);
  const [temp, setTemp] = useState(2);
  const [rain, setRain] = useState(-20);

  if (!data) return <Status loading={loading} error={error} />;
  const [sc, ood, cf] = data;

  const cell = (t, r) => {
    if (state === FED) {
      const f = sc.federation_mean.find((x) => x.delta_temp === t && x.rain_pct === r);
      return f ? { change: f.mean_pct_change, sd: null } : null;
    }
    const x = sc.responses.find((y) => y.client === state && y.delta_temp === t && y.rain_pct === r);
    return x ? { change: x.mean_pct_change, sd: x.sd_pct_change, base: x.mean_base_yield, pert: x.mean_pert_yield } : null;
  };
  const sel = cell(temp, rain);
  const all = TEMPS.flatMap((t) => RAINS.map((r) => cell(t, r)?.change ?? 0));
  const maxAbs = Math.max(...all.map(Math.abs), 1);

  const drop = ood.summary.map((r) => ({ ...r, name: armLabel(r.arm) }));
  const stateDrop = state === FED ? [] : ood.per_client.filter((r) => r.client === state);
  const deficit = state === FED ? null : ood.deficit_years?.[state];

  return (
    <>
      <PageHeader kicker="Climate conditions" title="How yield responds to heat and drought">
        Two questions. If the test years had been hotter or drier, how would predicted yield
        change? And do the models still hold up on each state's own drought years, which
        they never trained on?
      </PageHeader>

      <Controls>
        <Select id="cl-state" label="State" value={state} onChange={setState}
          options={[{ value: FED, label: `Federation average (${sc.clients.length} states)` },
            ...sc.clients.map((c) => ({ value: c, label: c }))]} />
        <Segmented label="Temperature" value={temp} onChange={setTemp}
          options={TEMPS.map((t) => ({ value: t, label: tLabel(t) }))} />
        <Segmented label="Rainfall" value={rain} onChange={setRain}
          options={RAINS.map((r) => ({ value: r, label: rLabel(r) }))} />
      </Controls>

      <Section title="Climate scenarios" tag={`${armLabel(sc.arm)} · ${sc.seeds.length} seeds`}
        lede="Temperature is raised and rainfall scaled in every month of 2012–2015, then the trained model predicts again. Only these nine combinations were computed, so the controls offer only these values.">
        <div className="grid-2">
          <Card>
            <div className="headline-stat">
              <div className={`v ${sel && sel.change < 0 ? "neg" : "pos"}`}>{sel ? fmt.pct(sel.change) : "—"}</div>
              <div className="l">change in predicted yield · temperature {tLabel(temp).toLowerCase()}, rainfall {rLabel(rain).toLowerCase()}</div>
              <div className="note">{state === FED ? `Mean across ${sc.clients.length} states` :
                sel?.sd != null ? `Spread across seeds ±${sel.sd.toFixed(1)} percentage points` : ""}</div>
            </div>
            {sel?.base != null && (
              <div style={{ marginTop: 12 }}>
                <KV k="Predicted yield, observed conditions" v={`${fmt.i(sel.base)} kg/ha`} />
                <KV k="Predicted yield, this scenario" v={`${fmt.i(sel.pert)} kg/ha`} />
              </div>
            )}
            <Callout warn title="Not a forecast.">
              This shows how the trained model responds to warmer or drier versions of years it
              has seen. The 1990–2015 data contains no emission pathway. A state whose yield
              rises under warming is a property of the model, not an agronomic finding.
            </Callout>
          </Card>

          <Card title={`All nine combinations · ${state === FED ? "federation average" : state}`}>
            <div className="table-wrap">
              <table className="heat">
                <thead>
                  <tr><th></th>{RAINS.map((r) => <th key={r}>Rain {rLabel(r)}</th>)}</tr>
                </thead>
                <tbody>
                  {TEMPS.map((t) => (
                    <tr key={t}>
                      <th className="row">{tLabel(t)}</th>
                      {RAINS.map((r) => {
                        const c = cell(t, r);
                        const on = t === temp && r === rain;
                        return (
                          <td key={r} className={on ? "sel" : ""}
                            style={{ background: divergingFill(c?.change, maxAbs) }}
                            onClick={() => { setTemp(t); setRain(r); }}
                            title="Select this combination">
                            {c ? fmt.pct(c.change) : "—"}
                            {c?.sd != null && c.sd > 0 && <small>±{c.sd.toFixed(1)}</small>}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="chart-note">Change in mean predicted yield against observed conditions. ± is the standard deviation across seeds. Select a cell to inspect it.</p>
          </Card>
        </div>
      </Section>

      <Section title="Drought robustness" tag="Out of distribution"
        lede="Models were retrained with each state's three driest monsoon years withheld, then tested on exactly those years. R² drop = R² on normal years minus R² on drought years; below zero means no loss.">
        <div className="grid-2">
          <Card title="R² drop on each state's own drought years">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={drop} layout="vertical" margin={{ top: 4, right: 50, bottom: 4, left: 8 }}>
                <CartesianGrid {...gridProps} horizontal={false} />
                <XAxis type="number" {...axisProps} tickFormatter={(v) => v.toFixed(2)} domain={[-0.25, 0.15]} />
                <YAxis type="category" dataKey="name" {...axisProps} width={96} />
                <ReferenceLine x={0} stroke={C.muted} />
                <Tooltip {...tooltipProps} formatter={(v, n, p) => [
                  `${fmt.sg(v)}  (95% CI ${fmt.n(p.payload.mean_r2_drop_ci_lo)} to ${fmt.n(p.payload.mean_r2_drop_ci_hi)})`, "R² drop"]} />
                <Bar dataKey="mean_r2_drop" isAnimationActive={false}
                  label={{ position: "right", fill: C.muted, fontSize: 11, formatter: (v) => fmt.sg(v) }}>
                  {drop.map((d) => <Cell key={d.arm} fill={d.mean_r2_drop <= 0 ? C.moss : C.clay} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Method</th><th>95% CI</th><th>Worst state</th></tr></thead>
                <tbody>
                  {drop.map((r) => (
                    <tr key={r.arm}>
                      <td className="name">{r.name}</td>
                      <td>[{fmt.n(r.mean_r2_drop_ci_lo)}, {fmt.n(r.mean_r2_drop_ci_hi)}]</td>
                      <td className="neg">{fmt.sg(r.worst_client_r2_drop, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Callout warn title="Read the negative values carefully.">
              They mean no measurable loss, not that models improve in drought. The normal-year
              reference covers few rows per state, and in every method at least one state's R²
              collapses on its drought years (worst state column).
            </Callout>
          </Card>

          <Card title={state === FED ? "Pick a state to see its drought years" : `${state} on its drought years`}>
            {state === FED ? (
              <p className="muted">Choose a state above to see which years were withheld and how each method did on them.</p>
            ) : (
              <>
                {deficit && <KV k="Withheld drought years" v={deficit.join(", ")} />}
                <div className="table-wrap" style={{ marginTop: 10 }}>
                  <table>
                    <thead><tr><th>Method</th><th>R² normal</th><th>R² drought</th><th>Drop</th></tr></thead>
                    <tbody>
                      {stateDrop.map((r) => (
                        <tr key={r.arm}>
                          <td className="name">{armLabel(r.arm)}</td>
                          <td>{fmt.n(r.r2_normal)}</td>
                          <td>{fmt.n(r.r2_ood)}</td>
                          <td className={r.r2_drop <= 0 ? "pos" : "neg"}>{fmt.sg(r.r2_drop)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="chart-note">R² within this state, averaged over seeds. Within one state R² is often low or negative; see the Yield prediction view for why.</p>
              </>
            )}
          </Card>
        </div>
      </Section>

      <Section title="Why the climate signal is hard to read" tag="Confound"
        lede={`The ${cf.n_total} month–variable pairs ranked by how far the correlation across all districts sits from the correlation within each district.`}>
        <Card>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Variable · month</th><th>Across districts r</th><th>Within district r</th><th></th></tr></thead>
              <tbody>
                {cf.rows.map((r) => (
                  <tr key={`${r.variable}-${r.month}`}>
                    <td className="name">{r.variable.replace(/_/g, " ")} · {r.month}</td>
                    <td>{fmt.sg(r.pooled_r)}</td>
                    <td>{fmt.sg(r.within_district_r)}</td>
                    <td>{r.sign_flip ? <span className="pill no">sign flips</span> : <span className="sd">weakens</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Callout title="Why the federation is posed by state.">
            Across all districts, post-monsoon rain looks helpful and monsoon rain looks harmful.
            Both are artefacts: wet districts tend to be rain-fed and low-yielding. Comparing each
            district only with itself reverses the sign. This is the heterogeneity the project
            keeps rather than averaging away.
          </Callout>
        </Card>
      </Section>

      <Provenance />
    </>
  );
}
