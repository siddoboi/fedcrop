import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useApi } from "../api.js";
import { armLabel, featureKind, fmt, pretty } from "../labels.js";
import { C, axisProps, gridProps, tooltipProps } from "../components/charts.jsx";
import AgreementChart from "../components/AgreementChart.jsx";
import {
  Callout, Card, Controls, KV, PageHeader, Pill, Provenance, Section, Select, Status,
} from "../components/ui.jsx";

const NATIONAL = "__national__";
const KIND_COLOR = { monsoon: C.moss, "post-monsoon": "#C9A55A", climate: "#7FA0B8", covariate: C.muted };
const KIND_LABEL = { monsoon: "Monsoon month", "post-monsoon": "Post-monsoon month", climate: "Other month", covariate: "Farm covariate" };
const ARM_ORDER = ["centralised", "fedper", "fedavg", "fedprox", "local"];

export default function Explainable() {
  const { data, error, loading } = useApi("/api/attributions", "/api/agreement", "/api/fidelity");
  const [arm, setArm] = useState(null);
  const [state, setState] = useState(NATIONAL);

  const attr = data?.[0];
  const arms = useMemo(() => {
    if (!attr) return [];
    const have = new Set([...Object.keys(attr.global || {}), ...Object.keys(attr.per_client || {})]);
    return ARM_ORDER.filter((a) => have.has(a));
  }, [attr]);

  if (!data) return <Status loading={loading} error={error} />;
  const [, ag, fid] = data;
  const a = arm ?? (arms.includes("fedper") ? "fedper" : arms[0]);
  const states = Object.keys(attr.per_client?.[a] || {}).sort();
  const st = state !== NATIONAL && states.includes(state) ? state : NATIONAL;
  const source = st === NATIONAL ? attr.global?.[a] : attr.per_client?.[a]?.[st];

  const entries = Object.entries(source || {}).map(([f, v]) => [f, Math.abs(Number(v))]);
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
  const top = entries.sort((x, y) => y[1] - x[1]).slice(0, 12)
    .map(([f, v]) => ({ name: pretty(f), feature: f, share: (v / total) * 100, kind: featureKind(f) }));

  const chk = ag.agronomic_check;

  return (
    <>
      <PageHeader kicker="Explainable AI" title="What the model weighs most heavily">
        GradientSHAP measures how much each input moves a prediction. Each state's
        attributions are computed against that state's own average, because in a federation
        no state may see another's data.
      </PageHeader>

      <Controls>
        <Select id="xai-arm" label="Method" value={a} onChange={setArm}
          options={arms.map((x) => ({ value: x, label: armLabel(x) }))} />
        <Select id="xai-state" label="Scope" value={st} onChange={setState}
          options={[{ value: NATIONAL, label: "National (all states)" },
            ...states.map((s) => ({ value: s, label: s }))]} />
      </Controls>

      <Section title={`Top 12 drivers · ${armLabel(a)} · ${st === NATIONAL ? "national" : st}`} tag="Attribution"
        lede="Share of the model's total attribution, out of 69 inputs (12 months × 5 climate variables, plus 9 farm covariates).">
        <Card>
          {top.length === 0 ? (
            <p className="muted">No attributions were computed for this method and scope.</p>
          ) : (
            <ResponsiveContainer width="100%" height={Math.max(300, top.length * 30)}>
              <BarChart data={top} layout="vertical" margin={{ top: 4, right: 40, bottom: 4, left: 8 }}>
                <CartesianGrid {...gridProps} horizontal={false} />
                <XAxis type="number" {...axisProps} tickFormatter={(v) => `${v.toFixed(0)}%`} />
                <YAxis type="category" dataKey="name" {...axisProps} width={210} interval={0} />
                <Tooltip {...tooltipProps}
                  formatter={(v, n, p) => [`${Number(v).toFixed(1)}% of attribution`, KIND_LABEL[p.payload.kind]]} />
                <Bar dataKey="share" isAnimationActive={false}
                  label={{ position: "right", fill: C.muted, fontSize: 11, formatter: (v) => `${v.toFixed(1)}%` }}>
                  {top.map((d) => <Cell key={d.feature} fill={KIND_COLOR[d.kind]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
          <div className="chips">
            {Object.entries(KIND_LABEL).map(([k, l]) => (
              <span key={k} className="chip"><span style={{ color: KIND_COLOR[k] }}>■</span> {l}</span>
            ))}
          </div>
        </Card>
      </Section>

      <Section title="Do federated explanations agree with centralised ones?" tag="Validation"
        lede="Rank agreement (Kendall τ) between each method's attributions and the centralised model's. The control is two centralised models differing only in random seed, so it marks the most agreement achievable at all.">
        <div className="grid-2-wide">
          <Card>
            <AgreementChart table={ag.table} />
            <Callout title="The gap is federation, not noise.">
              Seed variation alone leaves τ at {fmt.n(ag.table.find((r) => r.comparison.includes("seed control"))?.kendall_tau)}.
              Federated methods sit lower, and no federated interval overlaps the control.
              FedPer stays closest.
            </Callout>
          </Card>
          <Card>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <h3 style={{ margin: 0 }}>Agronomic plausibility</h3>
              <Pill ok={chk.verdict === "PASS"}>{chk.verdict}</Pill>
            </div>
            <p className="lede" style={{ fontSize: 13 }}>{chk.reason}, and no post-monsoon month appears in the top ten.</p>
            {chk.top_features.slice(0, 10).map((f, i) => (
              <KV key={f} k={`${i + 1}. ${pretty(f)}`}
                v={featureKind(f) === "monsoon" ? "monsoon" : ""} tone="pos" />
            ))}
            <div style={{ borderTop: `1px solid ${C.line}`, marginTop: 8, paddingTop: 6 }}>
              <KV k="Attribution share, monsoon months" v={`${(chk.share_monsoon * 100).toFixed(1)}%`} />
              <KV k="Attribution share, post-monsoon" v={`${(chk.share_post_monsoon * 100).toFixed(1)}%`} />
              <KV k="Attribution share, farm covariates" v={`${(chk.share_covariates * 100).toFixed(1)}%`} />
            </div>
          </Card>
        </div>
      </Section>

      <Section title="Do the explanations reflect what the model uses?" tag="Fidelity"
        lede="Delete the k highest-ranked inputs and measure how far the prediction moves (AOPC), then do the same with k random inputs. Real attributions move it more.">
        <Card>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Method</th><th>State</th><th>AOPC, top-ranked</th><th>AOPC, random</th><th>Ratio</th><th>Verdict</th></tr></thead>
              <tbody>
                {fid.map((r) => (
                  <tr key={`${r.arm}-${r.client}`}>
                    <td className="name">{armLabel(r.arm)}</td>
                    <td>{r.client}</td>
                    <td>{fmt.n(r.aopc_topk)}</td>
                    <td>{fmt.n(r.aopc_random)}</td>
                    <td>{r.ratio.toFixed(1)}×</td>
                    <td><Pill ok={r.verdict === "PASS"}>{r.verdict}</Pill></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Callout warn title="Do not rank methods by ratio.">
            FedAvg and FedProx have the highest ratios only because their models barely react
            to any deletion (random AOPC near 0.04). A ratio is comparable only between models
            that respond similarly at baseline.
          </Callout>
        </Card>
      </Section>

      <Provenance />
    </>
  );
}
