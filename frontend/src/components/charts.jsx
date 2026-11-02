// Chart styling shared by every Recharts chart, so axes and tooltips look the
// same on every view.

export const C = {
  bg: "#1C2B1C", card: "#223522", band: "#2C5F2D", line: "#3C5A3C",
  text: "#D8DED8", hi: "#FFFFFF", muted: "#A8B4A8", moss: "#97BC62", clay: "#B85042",
};

export const axisProps = {
  stroke: C.line,
  tick: { fill: C.muted, fontSize: 12 },
  tickLine: { stroke: C.line },
};

export const gridProps = { stroke: C.line, strokeDasharray: "3 3", strokeOpacity: 0.6 };

export const tooltipProps = {
  contentStyle: { background: C.bg, border: `1px solid ${C.line}`, borderRadius: 6, color: C.text },
  labelStyle: { color: C.hi, fontWeight: 600 },
  itemStyle: { color: C.text },
  cursor: { fill: "rgba(151,188,98,0.08)" },
};

export const legendProps = { wrapperStyle: { color: C.text, fontSize: 13 } };

/** Diverging bar centred on zero, from the original dashboard. */
export function SkillBar({ value, max, width = 150 }) {
  if (value == null) return null;
  const half = width / 2;
  const w = Math.max(2, (Math.abs(value) / max) * half);
  const pos = value >= 0;
  return (
    <span className="skillbar" style={{ width }}>
      <span className="zero" style={{ left: half }} />
      <i style={{ left: pos ? half : half - w, width: w, background: pos ? C.moss : C.clay }} />
    </span>
  );
}

/** Background colour for a % change: green up, clay down, strength by size. */
export function divergingFill(v, max) {
  if (v == null || Number.isNaN(v)) return "transparent";
  const a = Math.min(1, Math.abs(v) / (max || 1)) * 0.75 + 0.08;
  return v >= 0 ? `rgba(151,188,98,${a})` : `rgba(184,80,66,${a})`;
}
