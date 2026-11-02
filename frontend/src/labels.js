// Plain-language names, carried over from the original public page.
// Every view that shows a feature name goes through pretty().

export const ARM_LABELS = {
  global_mean: "Global mean",
  district_mean: "District mean",
  district_trend: "District trend",
  centralised: "Centralised",
  local: "Local-only",
  fedavg: "FedAvg",
  fedprox: "FedProx",
  fedper: "FedPer",
};
export const LEARNED_ARMS = ["centralised", "local", "fedper", "fedavg", "fedprox"];
export const FED_ARMS = ["fedavg", "fedprox", "fedper"];

export const ARM_COLORS = {
  centralised: "#D8DED8",
  local: "#C9A55A",
  fedper: "#97BC62",
  fedavg: "#B85042",
  fedprox: "#D98B6F",
};

export const armLabel = (a) => ARM_LABELS[a] || a;

const MONTHS = {
  jan: "January", feb: "February", mar: "March", apr: "April", may: "May",
  jun: "June", jul: "July", aug: "August", sep: "September", oct: "October",
  nov: "November", dec: "December",
};
const VARS = {
  max_temp: "maximum temperature", min_temp: "minimum temperature",
  precipitation: "rainfall", evapotranspiration: "evapotranspiration",
  windspeed: "wind speed",
};
const PLAIN = {
  irrigation_ratio: "Share of land irrigated",
  gross_irrigated_area: "Irrigated area",
  gross_cropped_area: "Total cropped area",
  nitrogen_consumption: "Nitrogen fertiliser use",
  phosphate_consumption: "Phosphate fertiliser use",
  potash_consumption: "Potash fertiliser use",
  fertiliser_per_ha: "Fertiliser per hectare",
  rice_area: "Area under rice",
  rice_area_share: "Rice share of cropped area",
};

export function pretty(f) {
  if (PLAIN[f]) return PLAIN[f];
  const m = /^([a-z]{3})_(.+)$/.exec(f);
  if (m && MONTHS[m[1]] && VARS[m[2]]) return `${MONTHS[m[1]]} ${VARS[m[2]]}`;
  return f.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

/** Which family a feature belongs to, for colouring bars. */
export function featureKind(f) {
  if (PLAIN[f]) return "covariate";
  if (/^(jun|jul|aug|sep)_/.test(f)) return "monsoon";
  if (/^(oct|nov|dec)_/.test(f)) return "post-monsoon";
  return "climate";
}

// Number formatting. en-US grouping: 294,912 reads better here than 2,94,912.
export const fmt = {
  n: (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d)),
  sg: (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d)),
  i: (v) => (v == null ? "—" : Math.round(v).toLocaleString("en-US")),
  pct: (v, d = 1) => (v == null ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d) + "%"),
};
