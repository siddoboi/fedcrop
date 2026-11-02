import { C } from "./charts.jsx";

/**
 * Kendall τ per comparison, with 95% interval whiskers and a dashed line at the
 * lower end of the seed-control interval. Hand-drawn SVG, ported from the
 * original dashboard: Recharts cannot draw the interval and control floor cleanly.
 * Bars live in 0..W; labels have their own column, so nothing collides.
 */
export default function AgreementChart({ table }) {
  const ctrl = table.find((r) => r.comparison.includes("seed control"));
  const rows = ctrl ? [ctrl, ...table.filter((r) => r !== ctrl)] : table;
  const max = 0.85, W = 260, LBL = W + 18, ROW = 54, VB = LBL + 190;
  const x = (v) => (v / max) * W;
  const name = (r) =>
    r.comparison.replace("centralised vs ", "").replace(" (seed control)", "");

  return (
    <svg viewBox={`0 0 ${VB} ${rows.length * ROW + 6}`} width="100%"
         role="img" aria-label="Kendall tau agreement with centralised explanations">
      {ctrl && (
        <g>
          <line x1={x(ctrl.kendall_tau_lo)} x2={x(ctrl.kendall_tau_lo)} y1={14}
                y2={rows.length * ROW - 4} stroke={C.hi} strokeWidth="1"
                strokeDasharray="3 3" opacity=".45" />
          <text x={x(ctrl.kendall_tau_lo) + 5} y={rows.length * ROW - 2}
                fontSize="10" fill={C.muted}>control floor</text>
        </g>
      )}
      {rows.map((r, i) => {
        const isCtrl = r === ctrl;
        const label = isCtrl ? "Centralised vs itself" : name(r);
        return (
          <g key={r.comparison} transform={`translate(0,${i * ROW})`}>
            <text x="0" y="12" fontSize="12.5" fill={isCtrl ? C.hi : C.text}
                  fontWeight={isCtrl ? 700 : 400}>
              {label === "fedavg" ? "FedAvg" : label === "fedprox" ? "FedProx"
                : label === "fedper" ? "FedPer" : label}
              {isCtrl && <tspan fill={C.muted} fontWeight="400" dx="8">seed-stability control</tspan>}
            </text>
            <rect x="0" y="20" width={x(r.kendall_tau)} height="12" rx="2"
                  fill={isCtrl ? "#F5F5F5" : C.moss} opacity={isCtrl ? 1 : 0.9} />
            <line x1={x(r.kendall_tau_lo)} x2={x(r.kendall_tau_hi)} y1="38.5" y2="38.5"
                  stroke={C.muted} strokeWidth="1.4" />
            <line x1={x(r.kendall_tau_lo)} x2={x(r.kendall_tau_lo)} y1="35" y2="42"
                  stroke={C.muted} strokeWidth="1.4" />
            <line x1={x(r.kendall_tau_hi)} x2={x(r.kendall_tau_hi)} y1="35" y2="42"
                  stroke={C.muted} strokeWidth="1.4" />
            <text x={LBL} y="31" fontSize="13" fill={C.hi} fontWeight="600">
              τ {r.kendall_tau.toFixed(3)}
            </text>
            <text x={LBL} y="45" fontSize="10.5" fill={C.muted}>
              95% CI [{r.kendall_tau_lo.toFixed(3)}, {r.kendall_tau_hi.toFixed(3)}]
            </text>
          </g>
        );
      })}
    </svg>
  );
}
