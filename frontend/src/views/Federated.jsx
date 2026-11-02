import { useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useApi } from "../api.js";
import { ARM_COLORS, FED_ARMS, armLabel, fmt } from "../labels.js";
import { C, axisProps, gridProps, legendProps, tooltipProps } from "../components/charts.jsx";
import {
  Callout, Card, Controls, PageHeader, Provenance, Section, Segmented, Select, Status,
} from "../components/ui.jsx";

export default function Federated() {
  const { data, error, loading } = useApi(
    "/api/federation", "/api/complexity", "/api/mu-sweep", "/api/significance");
  const [seed, setSeed] = useState(null);
  const [arm, setArm] = useState("fedper");

  const parsed = useMemo(() => {
    if (!data) return null;
    const [fed] = data;
    const arms = FED_ARMS.filter((a) => fed[a]);
    const seeds = [...new Set(arms.flatMap((a) => fed[a].map((r) => r.seed)))].sort((a, b) => a - b);
    return { fed, arms, seeds };
  }, [data]);

  if (!data) return <Status loading={loading} error={error} />;
  const [, cx, mu, sig] = data;
  const { fed, arms, seeds } = parsed;
  const s = seed ?? seeds[0];
  const run = (a) => fed[a].find((r) => r.seed === s);

  // One row per round, one column per method, for the weighted-mean chart.
  const maxRounds = Math.max(...arms.map((a) => run(a)?.history.length || 0));
  const weighted = Array.from({ length: maxRounds }, (_, i) => {
    const row = { round: i };
    arms.forEach((a) => { const h = run(a)?.history[i]; if (h) row[a] = h.val_rmse_weighted; });
    return row;
  });
  const fair = (run(arm)?.history || []).map((h) => ({
    round: h.round, mean: h.val_rmse_weighted, worst: h.val_rmse_worst, spread: h.val_rmse_std,
  }));
  const best = run(arm)?.best_round;

  const wil = sig.pairwise_wilcoxon || [];
  const ci = sig.ablation_with_ci || [];

  return (
    <>
      <PageHeader kicker="Federated learning" title="How the states train together">
        Each round, every state trains locally for 2 epochs and sends its weights to the
        server, which averages them. FedPer averages only the shared recurrent layer and
        leaves each state's final layer at home.
      </PageHeader>

      <Controls>
        <Segmented label="Seed" value={s} onChange={setSeed}
          options={seeds.map((v) => ({ value: v, label: String(v) }))} />
        <Select id="fl-arm" label="Fairness chart method" value={arm} onChange={setArm}
          options={arms.map((a) => ({ value: a, label: armLabel(a) }))} />
      </Controls>

      <Section title="Validation error by round" tag="Training"
        lede="Lower is better. Training stops 20 rounds after the best round and restores the best weights, so the curve continues past the point that is reported.">
        <div className="grid-2">
          <Card title="All three methods, weighted across states">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={weighted} margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="round" {...axisProps}
                  label={{ value: "round", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 12 }} />
                <YAxis {...axisProps} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} width={46} />
                <Tooltip {...tooltipProps} formatter={(v, n) => [Number(v).toFixed(4), armLabel(n)]} labelFormatter={(r) => `Round ${r}`} />
                <Legend {...legendProps} verticalAlign="top" height={30} formatter={armLabel} />
                {arms.map((a) => (
                  <Line key={a} type="monotone" dataKey={a} stroke={ARM_COLORS[a]} dot={false} strokeWidth={2} connectNulls={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
            <p className="chart-note">Validation RMSE in scaled units (yield standardised per state), not kg/ha.</p>
          </Card>

          <Card title={`${armLabel(arm)}: does the worst-off state improve?`}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={fair} margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="round" {...axisProps}
                  label={{ value: "round", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 12 }} />
                <YAxis {...axisProps} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} width={46} />
                <Tooltip {...tooltipProps} formatter={(v, n) => [Number(v).toFixed(4), n]} labelFormatter={(r) => `Round ${r}`} />
                <Legend {...legendProps} verticalAlign="top" height={30} />
                {best != null && <ReferenceLine x={best} stroke={C.hi} strokeDasharray="4 3"
                  label={{ value: `best round ${best}`, fill: C.muted, fontSize: 11, position: "insideTopRight" }} />}
                <Line type="monotone" dataKey="worst" name="Worst state" stroke={C.clay} dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="mean" name="Weighted mean" stroke={C.moss} dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="spread" name="Spread across states (sd)" stroke={C.muted} dot={false} strokeDasharray="5 4" />
              </LineChart>
            </ResponsiveContainer>
            <p className="chart-note">Scaled units. The gap between the worst state and the mean is the heterogeneity the project is about.</p>
          </Card>
        </div>
      </Section>

      <Section title="Communication cost" tag="System performance"
        lede="Per-round cost depends only on how many parameters are sent, not on how much data each state holds.">
        <Card>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Method</th><th>Rounds</th><th>KB per round</th><th>Total MB</th><th>s per round</th></tr></thead>
              <tbody>
                {cx.communication.map((c) => (
                  <tr key={c.algorithm} className={c.algorithm === "fedper" ? "hl" : ""}>
                    <td className="name">{armLabel(c.algorithm)}</td>
                    <td>{c.rounds_run}</td>
                    <td>{c.kb_per_round.toFixed(1)}</td>
                    <td>{c.total_mb.toFixed(1)}</td>
                    <td>{c.s_per_round.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Callout title="FedPer is the cheapest per round and the dearest overall.">
            It sends {cx.model.fedper_shared_pct}% of the model each round because each state's
            head stays local, but it needs more rounds to converge, so its total transfer is
            higher. Per-round cost matters on a slow rural link; total transfer matters for
            training time.
          </Callout>
        </Card>
      </Section>

      <div className="grid-2">
        <Section title="FedProx proximal weight" tag="Why federation loses">
          <Card>
            <div className="table-wrap">
              <table>
                <thead><tr><th style={{ textTransform: "none" }}>μ</th><th>R²</th><th>Skill</th><th>Drift from FedAvg</th><th>Rounds</th></tr></thead>
                <tbody>
                  {mu.map((m) => (
                    <tr key={m.mu}>
                      <td className="name">{m.mu === 0 ? "0 (FedAvg)" : m.mu}</td>
                      <td>{fmt.n(m.r2)}</td>
                      <td className={m.skill_vs_trend >= 0 ? "pos" : "neg"}>{fmt.sg(m.skill_vs_trend)}</td>
                      <td>{m.param_l2_vs_fedavg.toFixed(2)}</td>
                      <td>{m.rounds}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Callout warn title="The correction works, and makes things worse.">
              Pulling each state toward the shared model moves the solution further from
              FedAvg as μ grows, yet skill falls every time. The states need different models;
              constraining them hurts and personalising them helps. Single seed (42).
            </Callout>
          </Card>
        </Section>

        <Section title="Is the difference real?" tag="Statistics">
          <Card>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Method vs centralised</th><th>p-value</th><th>Result</th></tr></thead>
                <tbody>
                  {wil.map((w) => {
                    const sigf = w["significant_at_0.05"];
                    return (
                      <tr key={w.arm} className={w.arm === "fedper" ? "hl" : ""}>
                        <td className="name">{armLabel(w.arm)}</td>
                        <td>{w.p_value < 0.001 ? w.p_value.toFixed(4) : w.p_value.toFixed(3)}</td>
                        <td className={sigf ? "neg" : "muted"}>{sigf ? "significantly worse" : "no difference"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {ci.length > 0 && (
              <div className="table-wrap" style={{ marginTop: 18 }}>
                <table>
                  <thead><tr><th>Method</th><th>Skill</th><th>95% CI</th></tr></thead>
                  <tbody>
                    {ci.map((r) => (
                      <tr key={r.arm}>
                        <td className="name">{armLabel(r.arm)}</td>
                        <td className={r.skill_vs_trend_mean >= 0 ? "pos" : "neg"}>{fmt.sg(r.skill_vs_trend_mean)}</td>
                        <td>[{fmt.n(r.skill_vs_trend_ci_lo)}, {fmt.n(r.skill_vs_trend_ci_hi)}]</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <p className="chart-note">Wilcoxon signed-rank, paired by state on 2012–2015 RMSE averaged over 5 seeds. Intervals from 10,000 bootstrap resamples.</p>
          </Card>
        </Section>
      </div>

      <Provenance />
    </>
  );
}
