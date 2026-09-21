"""
End-to-end verification, written to docs/e2e_verification.md.

Runs the test suite, exercises every API route against a live server, checks
the screenshot set, and records what it found. The output is evidence for the
implementation-completeness section: it states what was actually executed and
when, rather than asserting that things work.

    uvicorn backend.app:app --port 8000 &
    python scripts/08_verify_e2e.py [--port 8000]

Exit code is non-zero if any check fails, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "e2e_verification.md"
SHOTS = ROOT / "docs" / "screenshots"

# route, expected status
ROUTES: list[tuple[str, int]] = [
    ("/api/health", 200),
    ("/api/headline", 200),
    ("/api/ablation", 200),
    ("/api/agreement", 200),
    ("/api/complexity", 200),
    ("/api/mu-sweep", 200),
    ("/api/fidelity", 200),
    ("/api/confound", 200),
    ("/api/clients", 200),
    ("/api/bundle", 200),
    ("/api/state/Uttar%20Pradesh", 200),
    ("/api/state/Nowhere", 404),        # not a client in the panel
    ("/api/results/meta", 200),
    ("/api/results/ood", 409),          # declared but not generated
    ("/api/results/nonsense", 404),     # not a declared artifact
    ("/", 200),
    ("/results", 200),
]

EXPECTED_SHOTS = [
    "public_full.png", "public_01_header.png", "public_02_profile.png",
    "public_03_drivers.png", "public_04_reliability.png",
    "dashboard_full.png", "01_header.png", "02_predictive.png",
    "03_explanation.png", "04_performance.png", "05_supporting.png",
    "06_status.png",
]


def probe(url: str) -> tuple[int, float, int]:
    """Return (status, milliseconds, bytes). HTTP errors are results, not faults."""
    t0 = time.perf_counter()
    try:
        with urlopen(url, timeout=20) as r:
            body = r.read()
            return r.status, (time.perf_counter() - t0) * 1000, len(body)
    except HTTPError as e:
        body = e.read()
        return e.code, (time.perf_counter() - t0) * 1000, len(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    failures: list[str] = []
    L: list[str] = []
    A = L.append

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    A("# End-to-end verification")
    A("")
    A(f"Run {stamp} · Python {platform.python_version()} · {platform.system()} "
      f"{platform.machine()}")
    A("")
    A("Produced by `scripts/08_verify_e2e.py`. Every line below is the recorded "
      "result of an executed check.")
    A("")

    # ---------------------------------------------------------------- tests
    A("## 1. Test suite")
    A("")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
    summary = tail[0] if tail else "no output"
    A("```")
    A("$ python -m pytest tests -q")
    A(summary)
    A("```")
    A("")
    if proc.returncode != 0:
        failures.append(f"test suite exited {proc.returncode}")
        A(f"**FAILED** (exit {proc.returncode}).")
        A("")
    else:
        for f, label in [("tests/test_pipeline.py", "pipeline"),
                         ("tests/test_api.py", "API")]:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", f, "-q", "--no-header"],
                cwd=ROOT, capture_output=True, text=True,
            )
            last = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1:]
            A(f"- {label}: {last[0] if last else '?'}")
        A("")

    # --------------------------------------------------------- gate A
    A("## 2. Pipeline gate A")
    A("")
    raw_xls = next((ROOT / "data" / "raw").glob("*.xls"), None)
    if raw_xls is None:
        A("Skipped: no raw dataset in `data/raw/`. Gate A checks the source panel "
          "against documented constants and cannot run without it.")
        A("")
    else:
        g = subprocess.run(
            [sys.executable, "scripts/00_verify.py"],
            cwd=ROOT, capture_output=True, text=True,
        )
        verdict = [ln for ln in g.stdout.splitlines() if ln.startswith("GATE A")]
        A("```")
        A("$ python scripts/00_verify.py")
        A(verdict[0] if verdict else f"(exit {g.returncode})")
        A("```")
        A("")
        if g.returncode != 0:
            failures.append(f"gate A exited {g.returncode}")
        else:
            A("Gate A re-derives the row counts, the pooled and within-district "
              "correlations, the JJAS deficit figures, the fertiliser collinearity and "
              "the baseline R² values from the source `.xls` and compares each against "
              "the documented constant. It is the check that the reported numbers were "
              "not transcribed by hand.")
            A("")

    # ------------------------------------------------------------- routes
    A("## 3. API routes")
    A("")
    A("| Route | Expected | Got | ms | Bytes |")
    A("|---|---:|---:|---:|---:|")
    reachable = True
    for route, expected in ROUTES:
        try:
            status, ms, size = probe(base + route)
        except URLError as e:
            A(f"| `{route}` | {expected} | unreachable | — | — |")
            failures.append(f"{route} unreachable ({e.reason})")
            reachable = False
            continue
        mark = "" if status == expected else "  **MISMATCH**"
        A(f"| `{route}` | {expected} | {status}{mark} | {ms:.0f} | {size:,} |")
        if status != expected:
            failures.append(f"{route}: expected {expected}, got {status}")
    A("")

    if not reachable:
        A(f"The API was not reachable at {base}. Start it with "
          "`uvicorn backend.app:app --port %d`." % args.port)
        A("")

    # -------------------------------------------------------- health detail
    if reachable:
        A("## 4. Artifact inventory as reported by the service")
        A("")
        with urlopen(base + "/api/health", timeout=20) as r:
            health = json.load(r)
        A(f"Service status: **{health['status']}**")
        A("")
        A("| Artifact | Required | Present | KB |")
        A("|---|---|---|---:|")
        for name, s in health["artifacts"].items():
            A(f"| {name} | {'yes' if s['required'] else 'no'} | "
              f"{'yes' if s['present'] else '**no**'} | {s['bytes'] / 1024:.1f} |")
        A("")
        if health["missing_required"]:
            failures.append(f"missing required artifacts: {health['missing_required']}")
            A(f"**Missing required:** {', '.join(health['missing_required'])}")
            A("")
        if health["missing_optional"]:
            A(f"Missing optional (pipeline stages not run): "
              f"{', '.join(health['missing_optional'])}. The service reports these "
              f"rather than serving empty responses, and a direct request for one "
              f"returns 409.")
            A("")

    # --------------------------------------------------------- screenshots
    A("## 5. Screenshots")
    A("")
    A("| File | KB | Status |")
    A("|---|---:|---|")
    for name in EXPECTED_SHOTS:
        p = SHOTS / name
        if not p.exists():
            A(f"| {name} | — | **missing** |")
            failures.append(f"screenshot missing: {name}")
        elif p.stat().st_size < 10_000:
            A(f"| {name} | {p.stat().st_size / 1024:.0f} | **suspiciously small** |")
            failures.append(f"screenshot likely blank: {name}")
        else:
            A(f"| {name} | {p.stat().st_size / 1024:.0f} | ok |")
    A("")

    # -------------------------------------------------------------- verdict
    A("## Verdict")
    A("")
    if failures:
        A(f"**{len(failures)} check(s) failed.**")
        A("")
        for f in failures:
            A(f"- {f}")
    else:
        A("All checks passed: test suite green, every route returned its expected "
          "status, the service reported its own artifact inventory, and the full "
          "screenshot set is present and non-trivial.")
    A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"{len(failures)} failure(s)" if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
