// Small shared building blocks. Kept in one file because each is a few lines.

export function PageHeader({ kicker, title, children }) {
  return (
    <header className="page-head">
      <p className="kicker">{kicker}</p>
      <h1>{title}</h1>
      {children && <p className="lede">{children}</p>}
    </header>
  );
}

export function Section({ title, tag, lede, children }) {
  return (
    <section className="section">
      <h2>{title}{tag && <span className="tag">{tag}</span>}</h2>
      {lede && <p className="lede">{lede}</p>}
      {children}
    </section>
  );
}

export function Card({ title, children, className = "" }) {
  return (
    <div className={`card ${className}`}>
      {title && <h3>{title}</h3>}
      {children}
    </div>
  );
}

export function Callout({ title, warn, children }) {
  return (
    <div className={`callout${warn ? " warn" : ""}`}>
      {title && <strong>{title} </strong>}
      {children}
    </div>
  );
}

export function Stat({ value, label, note, tone }) {
  return (
    <div className="stat">
      <div className={`v ${tone || ""}`}>{value}</div>
      <div className="l">{label}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export function KV({ k, v, tone }) {
  return (
    <div className="kv">
      <span>{k}</span>
      <span className={tone || ""}>{v}</span>
    </div>
  );
}

export function Pill({ ok, children }) {
  return <span className={`pill ${ok ? "ok" : "no"}`}>{children}</span>;
}

/** Loading and error states, so a dead API never renders as an empty page. */
export function Status({ loading, error }) {
  if (error) {
    return (
      <div className="status error">
        <strong>Could not reach the results service.</strong>
        <span>{error.message}</span>
        <span className="hint">
          Start it from the repository root with <code>uvicorn backend.app:app --port 8000</code>,
          then reload this page.
        </span>
      </div>
    );
  }
  if (loading) return <div className="status">Loading results…</div>;
  return null;
}

export function Select({ id, label, value, onChange, options }) {
  return (
    <label className="select" htmlFor={id}>
      <span>{label}</span>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

/** A small set of fixed choices, for inputs that only have a few valid values. */
export function Segmented({ label, value, onChange, options }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      <span>{label}</span>
      <div>
        {options.map((o) => (
          <button key={o.value} type="button"
            className={o.value === value ? "on" : ""}
            aria-pressed={o.value === value}
            onClick={() => onChange(o.value)}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Controls({ children }) {
  return <div className="controls">{children}</div>;
}

export function Provenance() {
  return (
    <p className="provenance">
      Trained 1990–2009 · validated 2010–2011 · tested 2012–2015 · 5 seeds (42–46) ·
      19 of 20 states federate (Telangana has no training years) · results computed
      offline and served read-only
    </p>
  );
}
