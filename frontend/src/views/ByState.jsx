import { useEffect, useState } from "react";
import { useApi } from "../api.js";
import { armLabel, fmt, pretty } from "../labels.js";
import { divergingFill } from "../components/charts.jsx";
import { Card, Controls, KV, PageHeader, Provenance, Section, Select, Stat, Status } from "../components/ui.jsx";

// Ported from the original public page. Written for a reader who is not an
// ML specialist, so it avoids method jargon where it can.

export default function ByState() {
  const clients = useApi("/api/clients");
  const [state, setState] = useState(null);

  const names = clients.data ? clients.data.rows.map((r) => r.client).sort() : [];
  useEffect(() => {
    if (!state && names.length) setState(names.includes("Uttar Pradesh") ? "Uttar Pradesh" : names[0]);
  }, [names.length]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!clients.data) return <Status loading={clients.loading} error={clients.error} />;
  const totalRows = clients.data.rows.reduce((s, r) => s + r.rows, 0);

  return (
    <>
      <PageHeader kicker="Rice yield insights" title="What the model learned about your state">
        From district records between 1990 and 2015. Pick a state to see its yield
        profile, the conditions the model weighs most heavily there, and how it responds
        to hotter or drier years.
      </PageHeader>

      <Controls>
        <Select id="bs-state" label="State" value={state || ""} onChange={setState}
          options={names.map((n) => ({ value: n, label: n }))} />
        <span className="muted" style={{ fontSize: 13.5 }}>
          {names.length} states · {fmt.i(totalRows)} district-years
        </span>
      </Controls>

      {state && <StateBody state={state} />}
      <Provenance />
    </>
  );
}

function StateBody({ state }) {
  const { data: d, error, loading } = useApi(`/api/state/${encodeURIComponent(state)}`);
  if (!d) return <Status loading={loading} error={error} />;
  const p = d.profile;

  return (
    <>
      <section className="section">
        <div className="stats" style={{ marginTop: 0 }}>
          <Stat value={fmt.i(p.mean_yield)} label="kg per hectare, average"
            note={`Ranked ${p.yield_rank} of ${p.n_states} states`} />
          <Stat value={fmt.i(p.rows)} label="district-years on record"
            note={`${d.record.year_min}–${d.record.year_max}`} />
          <Stat value={p.deficit_years.length} label="driest monsoon years"
            note={p.deficit_years.join(", ")} />
        </div>
      </section>

      <Drivers d={d} />
      <ClimateResponse d={d} />
      <Reliability d={d} />

      <section className="section">
        <div className="notice">
          <b>This page reports, it does not forecast.</b> Everything here is measured from
          district records for {d.record.year_min}–{d.record.year_max} and from models already
          trained on them. No prediction is made for a future season, and nothing here should
          be read as advice for one.
          {d.driver_source && <> The driver ranking comes from the <b>{armLabel(d.driver_source)}</b> model.</>}
        </div>
      </section>
    </>
  );
}

function Drivers({ d }) {
  if (!d.attribution_available) {
    return (
      <Section title="What drives yield here"
        lede={`No attribution was computed for ${d.state}. It has no records in the training years after cleaning, so it could not train its own model and is not part of the federation.`} />
    );
  }
  const max = Math.max(...d.drivers.map((x) => x.attribution));
  const local = new Set(d.drivers.map((x) => x.feature));
  const shared = d.national_top_features.filter((f) => local.has(f)).length;

  return (
    <Section title="What drives yield here"
      lede={`The conditions the model weighs most heavily for ${d.state}, as a share of its total attribution. A longer bar means the prediction moves more when that condition changes.`}>
      <div className="grid-2-wide">
        <Card>
          {d.drivers.map((x) => (
            <div className="fbar" key={x.feature}>
              <div>{pretty(x.feature)}</div>
              <div className="track"><div className="fill" style={{ width: `${(x.attribution / max) * 100}%` }} /></div>
              <div className="p">{(x.share * 100).toFixed(1)}%</div>
            </div>
          ))}
        </Card>
        <Card title="Compared with the national picture">
          <p className="lede" style={{ fontSize: 13.5 }}>
            Highlighted drivers also appear in {d.state}'s own top eight. {shared} of {d.national_top_features.length} overlap.
          </p>
          <div className="chips">
            {d.national_top_features.map((f) => (
              <span key={f} className={`chip ${local.has(f) ? "same" : ""}`}>{pretty(f)}</span>
            ))}
          </div>
          <p className="lede" style={{ fontSize: 13.5, marginTop: 14, marginBottom: 0 }}>
            Where they differ is the point of training by state: one national model would
            flatten these local differences into a single ranking.
          </p>
        </Card>
      </div>
    </Section>
  );
}

function ClimateResponse({ d }) {
  if (!d.climate_available) return null;
  const temps = [0, 1, 2], rains = [-20, 0, 20];
  const get = (t, r) => d.climate_response.find((x) => x.delta_temp === t && x.rain_pct === r);
  const maxAbs = Math.max(...d.climate_response.map((x) => Math.abs(x.mean_pct_change)), 1);
  const worst = d.climate_response.reduce((w, x) => (x.mean_pct_change < w.mean_pct_change ? x : w));

  return (
    <Section title="If the years had been hotter or drier"
      lede={`How the model's predicted yield for ${d.state} changes when 2012–2015 is made warmer or drier. The largest drop is ${fmt.pct(worst.mean_pct_change)}, ${worst.delta_temp === 0 ? "at observed temperature" : `at +${worst.delta_temp} °C`} with rainfall ${worst.rain_pct === 0 ? "as observed" : `${worst.rain_pct > 0 ? "+" : "−"}${Math.abs(worst.rain_pct)}%`}.`}>
      <Card>
        <div className="table-wrap">
          <table className="heat">
            <thead><tr><th></th>{rains.map((r) => <th key={r}>Rain {r === 0 ? "as observed" : `${r > 0 ? "+" : "−"}${Math.abs(r)}%`}</th>)}</tr></thead>
            <tbody>
              {temps.map((t) => (
                <tr key={t}>
                  <th className="row">{t === 0 ? "Temperature as observed" : `+${t} °C`}</th>
                  {rains.map((r) => {
                    const c = get(t, r);
                    return (
                      <td key={r} style={{ background: divergingFill(c?.mean_pct_change, maxAbs), cursor: "default" }}>
                        {c ? fmt.pct(c.mean_pct_change) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="chart-note">A sensitivity test of the model, not a climate projection.</p>
      </Card>
    </Section>
  );
}

function Reliability({ d }) {
  const t = d.trend_baseline;
  if (!t) return null;
  return (
    <Section title="How predictable this state has been"
      lede="Measured against the simplest honest benchmark: continuing each district's own historical trend. It sets the bar any model has to clear.">
      <div className="grid-2">
        <Card>
          <KV k="Variance explained (R²)" v={fmt.n(t.r2)} tone={t.r2 < 0 ? "neg" : "pos"} />
          <KV k="Typical error (RMSE)" v={`${fmt.i(t.rmse)} kg/ha`} />
          <KV k="Average error (MAE)" v={`${fmt.i(t.mae)} kg/ha`} />
          <KV k="Test district-years" v={fmt.i(t.n)} />
        </Card>
        <Card>
          <p className="lede" style={{ margin: 0, fontSize: 13.5 }}>
            An R² near 1.0 means yield here followed a steady trend and was easy to anticipate.
            A low or negative value means yield moved year to year in ways a trend line cannot
            capture, usually because of rainfall, which is where a climate-aware model has
            something to add.
          </p>
          {t.r2 < 0 && (
            <p className="lede neg" style={{ margin: "12px 0 0", fontSize: 13.5 }}>
              A negative R² means the trend line did worse than simply predicting this state's
              average yield every year.
            </p>
          )}
        </Card>
      </div>
    </Section>
  );
}
