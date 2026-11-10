"""The Docker image serves the built React app from the API's own origin.

backend.app reads FEDCROP_FRONTEND_DIR at import time, so each case runs in a
fresh interpreter with its own environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]

PROBE = textwrap.dedent("""
    import json
    from fastapi.testclient import TestClient
    from backend.app import app
    c = TestClient(app)
    root = c.get("/")
    out = {"root_status": root.status_code, "root_type": root.headers.get("content-type", ""),
           "root_body": root.text[:200], "api_status": c.get("/api").status_code,
           "asset_status": c.get("/assets/app.js").status_code,
           "health_status": c.get("/api/health").status_code}
    print(json.dumps(out))
""")


def probe(frontend_dir: Path | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "FEDCROP_FRONTEND_DIR"}
    if frontend_dir is not None:
        env["FEDCROP_FRONTEND_DIR"] = str(frontend_dir)
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture()
def dist(tmp_path: Path) -> Path:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    return tmp_path


def test_without_the_variable_root_stays_the_json_index():
    r = probe(None)
    assert r["root_status"] == 200 and "application/json" in r["root_type"]
    assert r["asset_status"] == 404


def test_with_a_build_root_serves_the_app_and_api_is_untouched(dist):
    r = probe(dist)
    assert r["root_status"] == 200 and "text/html" in r["root_type"]
    assert "id=root" in r["root_body"]
    assert r["asset_status"] == 200
    assert r["api_status"] == 200          # route index moved to /api
    assert r["health_status"] == 200


def test_a_directory_without_index_html_is_ignored(tmp_path):
    r = probe(tmp_path)
    assert "application/json" in r["root_type"]
