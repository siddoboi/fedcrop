"""
Capture dashboard screenshots for the report.

Writes a full-page image plus one image per section, so the report can use a
single section rather than cropping the long page by hand.

    uvicorn backend.app:app --port 8000 &
    python scripts/07_screenshots.py [--port 8000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"

PUBLIC_SECTIONS = [
    ("header", "public_01_header"),
    ("section:nth-of-type(1)", "public_02_profile"),
    ("section:nth-of-type(2)", "public_03_drivers"),
    ("section:nth-of-type(3)", "public_04_reliability"),
]

SECTIONS = [
    ("header", "01_header"),
    ("section:nth-of-type(1)", "02_predictive"),
    ("section:nth-of-type(2)", "03_explanation"),
    ("section:nth-of-type(3)", "04_performance"),
    ("section:nth-of-type(4)", "05_supporting"),
    ("section:nth-of-type(5)", "06_status"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=1320)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:  pip install playwright && playwright install chromium")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": args.width, "height": 1000},
            device_scale_factor=args.scale,
        )
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        try:
            page.goto(url, wait_until="networkidle")
        except Exception as exc:
            print(f"could not reach {url}: {exc}")
            print("start the API first:  uvicorn backend.app:app --port", args.port)
            browser.close()
            return 1

        page.wait_for_timeout(1200)

        if errors:
            print("JavaScript errors on the page, screenshots will be wrong:")
            for e in errors:
                print("  -", e)
            browser.close()
            return 1

        page.screenshot(path=str(OUT / "public_full.png"), full_page=True)
        print("wrote public_full.png")
        for selector, name in PUBLIC_SECTIONS:
            el = page.query_selector(selector)
            if el is None:
                print(f"  [skip] {name}: no element matching {selector}")
                continue
            el.screenshot(path=str(OUT / f"{name}.png"))
            print(f"wrote {name}.png")

        try:
            page.goto(url + "results", wait_until="networkidle")
        except Exception as exc:
            print(f"could not reach {url}results: {exc}")
            browser.close()
            return 1
        page.wait_for_timeout(1200)
        if errors:
            print("JavaScript errors on the results page:")
            for e in errors:
                print("  -", e)
            browser.close()
            return 1

        page.screenshot(path=str(OUT / "dashboard_full.png"), full_page=True)
        print("wrote dashboard_full.png")

        for selector, name in SECTIONS:
            el = page.query_selector(selector)
            if el is None:
                print(f"  [skip] {name}: no element matching {selector}")
                continue
            el.screenshot(path=str(OUT / f"{name}.png"))
            print(f"wrote {name}.png")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
