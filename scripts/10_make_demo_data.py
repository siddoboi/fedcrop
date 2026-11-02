"""Demo CSV files for the "Predict from CSV" page.

    python scripts/10_make_demo_data.py

Every row comes from the cleaned panel's test years (2012-2015), which no model
saw in training. These files demonstrate the upload pathway on held-out data;
they are not new evidence, because the same rows are already in the test set.

Only rows whose three derived ratios can be recomputed exactly from the six
raw farm figures are used. Cleaning imputes all nine covariates independently,
so for a row with an imputed figure, recomputing a ratio gives a different
value from the stored one, and an upload of that row would not reproduce the
pipeline's prediction. Those rows are skipped and counted, and any federated
state left with no usable rows is named.

Writes demo_data/*.csv and demo_data/README.md at the repo root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedcrop import columns as C  # noqa: E402
from fedcrop import io_layer  # noqa: E402
from fedcrop.cleaning import add_derived  # noqa: E402
from fedcrop.config import load_config  # noqa: E402

OUT = ROOT / "demo_data"
SINGLE_STATES = ["Jharkhand", "Punjab", "Kerala", "Maharashtra"]
PER_STATE_DISTRICTS = 2        # for the all-states file, to stay well under the upload limits
UPLOAD_MAX_ROWS, UPLOAD_MAX_BYTES = 2000, 2 * 1024 * 1024   # backend/predict.py MAX_ROWS, MAX_BYTES
SEQ_SOURCE = C.sequential_columns()
SEQ_NAMES = C.feature_names(C.SEQUENTIAL_VARS, C.MONTHS_FULL, [])
# feature_names treats an empty list as "use the defaults", so slice instead of passing [].
COV_NAMES = C.feature_names(C.SEQUENTIAL_VARS, C.MONTHS_FULL, C.BASE_COVARIATES)[len(SEQ_SOURCE):]
MIXED_STATES = ["Punjab", "Kerala", "Maharashtra", "West Bengal"]


def to_template(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Source columns -> the upload template's canonical columns."""
    out = pd.DataFrame({
        "state": df[cfg.state_key].to_numpy(),
        "district_code": df[cfg.district_key].to_numpy(),
        "district_name": df["Dist Name"].to_numpy() if "Dist Name" in df else "",
        "year": df[cfg.year_key].astype(int).to_numpy(),
    })
    for s, c in zip(SEQ_SOURCE, SEQ_NAMES):
        out[c] = df[s].to_numpy()
    for s, c in zip(C.BASE_COVARIATES, COV_NAMES):
        out[c] = df[s].to_numpy()
    out["actual_yield"] = df[cfg.target].to_numpy()
    return out


def main() -> int:
    cfg = load_config()
    clean_path = cfg.path("interim").with_name("clean.parquet")
    if not clean_path.exists():
        print(f"no cleaned panel at {clean_path}. Run scripts/01_build_features.py first.")
        return 1
    scalers_path = cfg.path("scalers") / "scalers.json"
    if not scalers_path.exists():
        print("no scalers.json. Run scripts/03_train_all.py first.")
        return 1
    federated = set(json.loads(scalers_path.read_text(encoding="utf-8")))

    clean = io_layer.load_interim(clean_path)
    lo, hi = cfg.test_years
    test = clean[(clean[cfg.year_key] >= lo) & (clean[cfg.year_key] <= hi)].reset_index(drop=True)

    # Keep rows whose stored ratios equal the ratios recomputed from the raw
    # figures, i.e. rows with no imputed covariate.
    recomputed = add_derived(test[C.BASE_COVARIATES])
    same = np.ones(len(test), dtype=bool)
    for col in C.DERIVED_COVARIATES:
        same &= np.isclose(recomputed[col].to_numpy(dtype=float),
                           test[col].to_numpy(dtype=float), rtol=1e-9, atol=1e-12)
    usable = test[same & test[cfg.state_key].isin(federated)].reset_index(drop=True)
    skipped = int((~same).sum())
    print(f"test rows {len(test)}; skipped {skipped} with imputed farm figures; "
          f"{len(usable)} usable from {usable[cfg.state_key].nunique()} of {len(federated)} federated states")
    for st in sorted(federated - set(usable[cfg.state_key])):
        n = int((test[cfg.state_key] == st).sum())
        print(f"  no usable rows for {st}: {n} test rows, all with at least one imputed farm figure")

    OUT.mkdir(parents=True, exist_ok=True)
    index = []

    def write(name: str, frame: pd.DataFrame, title: str, description: str) -> None:
        if frame.empty:
            print(f"skipped {name}: no usable rows")
            return
        frame.to_csv(OUT / name, index=False)
        size = (OUT / name).stat().st_size
        index.append({"file": name, "title": title, "description": description, "rows": int(len(frame))})
        note = "" if len(frame) <= UPLOAD_MAX_ROWS and size <= UPLOAD_MAX_BYTES else "  (OVER THE UPLOAD LIMIT)"
        print(f"wrote {name}: {len(frame)} rows, {size / 1024:.0f} KB{note}")

    # 1-4. One state each.
    for st in SINGLE_STATES:
        rows = to_template(usable[usable[cfg.state_key] == st], cfg)
        lede = ("Includes actual yield, so accuracy is reported. FedPer predictions should match the "
                "project's frozen test predictions." if st == "Jharkhand" else "Includes actual yield.")
        write(f"demo_{st.lower().replace(' ', '_')}.csv", rows, f"{st}, 2012–2015",
              f"Every usable district-year from {st}. {lede}")

    # 5. A few districts from contrasting states.
    parts = []
    for st in MIXED_STATES:
        rows = usable[usable[cfg.state_key] == st]
        codes = sorted(rows[cfg.district_key].unique())[:3]
        parts.append(rows[rows[cfg.district_key].isin(codes)])
    mixed = to_template(pd.concat(parts, ignore_index=True), cfg)
    write("demo_mixed_states.csv", mixed, "Four contrasting states",
          "Three districts each from irrigated Punjab, high-rainfall Kerala, semi-arid Maharashtra "
          "and West Bengal. Each row is predicted with its own state's model.")

    # 6. Same rows without the answer.
    write("demo_no_yield.csv", mixed.drop(columns=["actual_yield"]), "Prediction only",
          "The four-state file with the actual_yield column removed: predictions without accuracy figures.")

    # 7. Every state with usable rows, a couple of districts each.
    parts = []
    for st in sorted(usable[cfg.state_key].unique()):
        rows = usable[usable[cfg.state_key] == st]
        codes = sorted(rows[cfg.district_key].unique())[:PER_STATE_DISTRICTS]
        parts.append(rows[rows[cfg.district_key].isin(codes)])
    every = to_template(pd.concat(parts, ignore_index=True), cfg)
    write("demo_all_states.csv", every, f"All {every['state'].nunique()} states with usable rows",
          f"{PER_STATE_DISTRICTS} districts from each federated state that has usable test rows, "
          "every test year. Shows the per-state models side by side.")

    # 8. Valid rows plus one deliberate problem each.
    good = mixed.head(6).copy()
    broken = []

    tel = clean[(clean[cfg.state_key] == "Telangana") & (clean[cfg.year_key] >= lo)]
    row = (to_template(tel.head(1), cfg) if len(tel) else mixed.head(1).assign(state="Telangana")).copy()
    broken.append(row)                                                    # no model for this state
    broken.append(mixed.iloc[[1]].assign(state="Punjb"))                  # misspelt state
    r = mixed.iloc[[2]].copy(); r["jul_precipitation"] = np.nan; broken.append(r)          # missing climate
    r = mixed.iloc[[3]].copy().astype({"nitrogen_consumption": object})
    r["nitrogen_consumption"] = "n/a"; broken.append(r)                                   # text in a number
    r = mixed.iloc[[4]].copy(); r["rice_area"] = 1.5; broken.append(r)    # below the 2k ha threshold
    r = mixed.iloc[[5]].copy(); r["jul_precipitation"] = r["jul_precipitation"] * 10
    broken.append(r)                                                      # accepted, with a range warning
    train_row = clean[(clean[cfg.state_key] == "Punjab") & (clean[cfg.year_key] == 2005)].head(1)
    if len(train_row):
        broken.append(to_template(train_row, cfg))                        # accepted, training-year notice

    errors = pd.concat([good] + broken, ignore_index=True)
    write("demo_with_errors.csv", errors, "File with problems",
          "Six valid rows, then: a Telangana row (no model), a misspelt state, a missing July rainfall, "
          "text in a number column, a rice area below 2,000 ha, an extreme July rainfall (accepted with a "
          "warning) and a 2005 row (accepted, flagged as a training year).")

    lines = ["# Demo files for Predict from CSV", "",
             "Generated by `python scripts/10_make_demo_data.py` from the 2012–2015 test years, which no "
             "model saw in training. They show the upload working on held-out data. They are not new "
             "evidence: the same rows are already in the project's test set.", "",
             f"Rows with an imputed farm figure are left out ({skipped} of {len(test)} test rows), because "
             "their stored ratios cannot be recomputed from the raw figures.", "",
             "To try a file, open the app's Predict from CSV page and upload it.", "",
             "| File | Rows | What it shows |", "|---|---|---|"]
    lines += [f"| `{d['file']}` | {d['rows']} | {d['description']} |" for d in index]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
