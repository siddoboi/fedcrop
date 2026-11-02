import { useApi } from "../api.js";
import { fmt } from "../labels.js";
import { SkillBar } from "../components/charts.jsx";
import {
  Callout, Card, PageHeader, Pill, Provenance, Section, Stat, Status,
} from "../components/ui.jsx";

export default function Overview() {
  const { data, error, loading } = useApi("/api/bundle");
  if (!data) return <Status loading={loading} error={error} />;

  const h = data.headline;
  const ab = data.ablation;
  const learned = ab.rows.filter((r) => r.skill_vs_trend != null);
  const max = Math.max(...learned.map((r) => Math.abs(r.skill_vs_trend)), 0.01);
  const health = data.health;

  return (
    <>
      <PageHeader kicker="Federated and explainable AI · crop yield analytics"
        title="Predicting rice yield under climate change, without pooling the data">
        Rice yield across {h.districts} Indian districts, {h.year_min}–{h.year_max}. Each
        state trains on its own records and shares only model weights. This dashboard
        reports results computed offline and frozen; nothing here is re-estimated live.
      </PageHeader>

      <div className="stats" style={{ marginTop: 0, marginBottom: 36 }}>
        <Stat value={fmt.i(h.rows)} label="district-years after cleaning" />
        <Stat value={h.districts} label="districts" />
        <Stat value={`${h.clients_federated} of ${h.states}`} label="states federated" />
        <Stat value={fmt.n(h.best_r2)} label="best R², centralised" />
        <Stat value={fmt.sg(h.best_skill)} label="skill vs trend baseline" tone="pos" />
        <Stat value={h.seeds} label="seeds per method" />
      </div>

      <Section title="Accuracy against a simple baseline" tag="Results"
        lede={`Skill is 1 − MSE(model) / MSE(district trend): positive means the method beats
          extrapolating each district's own history. Baselines are deterministic; learned
          methods are the mean of ${ab.seeds.length} seeds (${ab.seeds.join(", ")}).`}>
        <Card>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Method</th><th>R²</th><th>Skill vs trend</th><th></th>
                  <th>RMSE (kg/ha)</th><th>Worst state RMSE</th><th>Spread across states</th><th>Seeds</th>
                </tr>
              </thead>
              <tbody>
                {ab.rows.map((r) => (
                  <tr key={r.arm} className={`${r.is_baseline ? "baseline" : ""} ${r.arm === "fedper" ? "hl" : ""}`}>
                    <td className="name">
                      {r.label}
                      {r.is_reference && <span className="ref-badge">reference</span>}
                    </td>
                    <td>
                      {fmt.n(r.r2)}
                      {r.n_seeds > 1 && <span className="sd"> ± {fmt.n(r.r2_sd)}</span>}
                    </td>
                    <td>
                      {r.skill_vs_trend == null ? <span className="sd">—</span> : (
                        <>
                          <span className={r.skill_vs_trend >= 0 ? "pos" : "neg"}>{fmt.sg(r.skill_vs_trend)}</span>
                          <span className="sd"> ± {fmt.n(r.skill_sd)}</span>
                        </>
                      )}
                    </td>
                    <td><SkillBar value={r.skill_vs_trend} max={max} /></td>
                    <td>{fmt.i(r.rmse)}</td>
                    <td>{fmt.i(r.worst_client_rmse)}</td>
                    <td>{fmt.i(r.cross_client_rmse_std)}</td>
                    <td className="sd">{r.n_seeds}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Callout title="Reading this table.">
            States differ so much in irrigation, fertiliser use and rainfall that one averaged
            model (FedAvg, FedProx) fits none of them well and falls below the trend line.
            FedPer shares the recurrent layer but keeps a small head per state, and recovers to
            baseline level while every state's data stays local. This is the expected ordering
            under strongly non-IID data, not a failed federation.
          </Callout>
        </Card>
      </Section>

      <Section title="What the service is serving" tag="Validation"
        lede="Every view reads these frozen result files. A file the pipeline has not produced is shown as missing rather than drawn as an empty chart.">
        <Card>
          <div className="grid-status">
            {Object.entries(health.artifacts).map(([k, s]) => (
              <div className="st" key={k}>
                <b>{k.replace(/_/g, " ")}</b>
                <Pill ok={s.present}>{s.present ? (s.required ? "built" : "optional") : "missing"}</Pill>
              </div>
            ))}
          </div>
          {health.missing_required.length > 0 ? (
            <Callout warn title="Required results are missing:">
              {health.missing_required.join(", ")}. Run the pipeline stage that produces them.
            </Callout>
          ) : (
            <Callout title="All required results present.">
              The API reports healthy only when every file the results freeze requires exists.
            </Callout>
          )}
        </Card>
      </Section>

      <Provenance />
    </>
  );
}
