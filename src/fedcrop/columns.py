"""Source column names.

This module exists because the dataset names months inconsistently:
temperature, precipitation and evapotranspiration use full month names,
windspeed uses abbreviations, and four of the twelve differ.

The source also misspells "precipitation" as "PERCIPITATION". That typo
is reproduced here, in exactly one place.
"""

from __future__ import annotations

MONTHS_FULL: list[str] = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
]

MONTH_SHORT: dict[str, str] = {
    "JANUARY": "jan", "FEBRUARY": "feb", "MARCH": "mar", "APRIL": "apr",
    "MAY": "may", "JUNE": "jun", "JULY": "jul", "AUGUST": "aug",
    "SEPTEMBER": "sep", "OCTOBER": "oct", "NOVEMBER": "nov", "DECEMBER": "dec",
}

# Windspeed columns use these abbreviations instead of the full month name.
# Hand-written, never derived: four of twelve differ from the full name.
WIND_MONTHS: dict[str, str] = {
    "JANUARY": "JAN",
    "FEBRUARY": "FEB",
    "MARCH": "MARCH",
    "APRIL": "APRIL",
    "MAY": "MAY",
    "JUNE": "JUNE",
    "JULY": "JULY",
    "AUGUST": "AUG",
    "SEPTEMBER": "SEPT",
    "OCTOBER": "OCT",
    "NOVEMBER": "NOV",
    "DECEMBER": "DEC",
}

# Canonical variable key -> (column suffix, uses windspeed month naming)
_VAR_SPEC: dict[str, tuple[str, bool]] = {
    "max_temp": ("MAXIMUM TEMPERATURE (Centigrate)", False),
    "min_temp": ("MINIMUM TEMPERATURE (Centigrate)", False),
    "precipitation": ("PERCIPITATION (Millimeters)", False),  # source typo
    "evapotranspiration": ("ACTUAL EVAPOTRANSPIRATION (Millimeters)", False),
    "windspeed": ("WINDSPEED (Meter per second)", True),
}

SEQUENTIAL_VARS: list[str] = list(_VAR_SPEC.keys())

TARGET = "RICE YIELD (Kg per ha)"

KEY_COLUMNS = ["Dist Code", "Dist Name", "State Code", "State Name", "Year"]

# Source covariate columns kept as model inputs.
BASE_COVARIATES = [
    "RICE AREA (1000 ha)",
    "NITROGEN CONSUMPTION (tons)",
    "PHOSPHATE CONSUMPTION (tons)",
    "POTASH CONSUMPTION (tons)",
    "GROSS CROPPED AREA (1000 ha)",
    "GROSS IRRIGATED AREA (1000 ha)",
]

# Ratios computed in cleaning.add_derived. Comparable across district sizes.
DERIVED_COVARIATES = ["irrigation_ratio", "fertiliser_per_ha", "rice_area_share"]

ANNUAL_COVARIATES = BASE_COVARIATES + DERIVED_COVARIATES

# Never model inputs.
LEAKAGE_COLUMNS = ["RICE PRODUCTION (1000 tons)"]
COLLINEAR_COLUMNS = ["TOTAL FERTILISER CONSUMPTION (tons)"]
HIGH_MISSING_COLUMNS = ["TOTAL AGRICULTURAL LABOUR POPULATION (1000 Number)"]

WINDOWS: dict[str, list[str]] = {
    "jan_dec": MONTHS_FULL,
    "jun_nov": ["JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER"],
}

JJAS = ["JUNE", "JULY", "AUGUST", "SEPTEMBER"]


def build_column_name(var: str, month: str) -> str:
    """Exact source column string for one variable in one month."""
    if var not in _VAR_SPEC:
        raise KeyError(f"unknown variable {var!r}; expected one of {SEQUENTIAL_VARS}")
    if month not in MONTHS_FULL:
        raise KeyError(f"unknown month {month!r}")
    suffix, uses_wind_naming = _VAR_SPEC[var]
    prefix = WIND_MONTHS[month] if uses_wind_naming else month
    return f"{prefix} {suffix}"


def sequential_columns(vars_: list[str] | None = None,
                       months: list[str] | None = None) -> list[str]:
    """Source columns for the sequence tensor, month-major then variable.

    The order here defines the flattened feature order and must match
    feature_names() index for index.
    """
    vars_ = vars_ or SEQUENTIAL_VARS
    months = months or MONTHS_FULL
    return [build_column_name(v, m) for m in months for v in vars_]


def precipitation_columns(months: list[str] | None = None) -> list[str]:
    months = months or MONTHS_FULL
    return [build_column_name("precipitation", m) for m in months]


def feature_names(vars_: list[str] | None = None,
                  months: list[str] | None = None,
                  covariates: list[str] | None = None) -> list[str]:
    """Readable names aligned index-for-index with the flattened feature vector.

    Every attribution label and every plot axis comes from this function.
    If SHAP labels are ever wrong, they are wrong here.
    """
    vars_ = vars_ or SEQUENTIAL_VARS
    months = months or MONTHS_FULL
    covariates = covariates if covariates is not None else ANNUAL_COVARIATES
    seq = [f"{MONTH_SHORT[m]}_{v}" for m in months for v in vars_]
    cov = [c.split(" (")[0].lower().replace(" ", "_") for c in covariates]
    return seq + cov


def validate_columns(df, expected: list[str]) -> None:
    """Assert every expected column exists, reporting all misses at once."""
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} expected column(s) missing from the frame:\n  "
            + "\n  ".join(missing)
        )
