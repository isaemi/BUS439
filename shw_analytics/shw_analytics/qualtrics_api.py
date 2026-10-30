"""Pull a full response export straight from Qualtrics (REST API v3) into the
exports folder, then rebuild the database. Needs an API token from a Qualtrics
account that can see the survey; UBC's Qualtrics admins decide whether API access
is enabled.

    python -m shw_analytics.qualtrics_api            # download + ingest
    python -m shw_analytics.qualtrics_api --no-ingest

Settings (in .env):
    QUALTRICS_API_TOKEN=...            never commit
    QUALTRICS_BASE_URL=https://ubc.ca1.qualtrics.com
    QUALTRICS_SURVEY_ID=SV_...

Each run downloads every response (not just new ones), so the latest file is
always complete; ingest de-duplicates by ResponseId across files anyway.
"""
import argparse
import datetime as dt
import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

from . import config

EXPORT_DIR = config.DB_PATH.parent / "exports"


def _request(method, url, token, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "X-API-TOKEN": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def export_responses(survey_id, token, base_url, out_dir=EXPORT_DIR, poll_seconds=2, timeout=600):
    """Start an export, wait for it, download the zip, save the CSV. Returns its path."""
    api = f"{base_url.rstrip('/')}/API/v3/surveys/{survey_id}/export-responses"
    # Same layout the pipeline expects: numeric codes, multi-select split into columns.
    body = {"format": "csv", "useLabels": False, "breakoutSets": True}
    progress_id = json.loads(_request("POST", api, token, body))["result"]["progressId"]
    waited = 0
    while True:
        status = json.loads(_request("GET", f"{api}/{progress_id}", token))["result"]
        if status["status"] == "complete":
            file_id = status["fileId"]
            break
        if status["status"] == "failed":
            raise RuntimeError(f"Qualtrics export failed: {status}")
        if waited > timeout:
            raise TimeoutError("Qualtrics export did not finish in time")
        time.sleep(poll_seconds)
        waited += poll_seconds
    archive = zipfile.ZipFile(io.BytesIO(_request("GET", f"{api}/{file_id}/file", token)))
    csv_name = next(n for n in archive.namelist() if n.endswith(".csv"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"api_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    path.write_bytes(archive.read(csv_name))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", default=os.getenv("QUALTRICS_SURVEY_ID"))
    ap.add_argument("--no-ingest", action="store_true")
    a = ap.parse_args()
    token, base = os.getenv("QUALTRICS_API_TOKEN"), os.getenv("QUALTRICS_BASE_URL")
    if not (token and base and a.survey):
        raise SystemExit("Set QUALTRICS_API_TOKEN, QUALTRICS_BASE_URL and QUALTRICS_SURVEY_ID in .env")
    path = export_responses(a.survey, token, base)
    print(f"Saved {path}")
    if not a.no_ingest:
        from . import ingest
        tables = ingest.load(EXPORT_DIR)
        print(f"Database rebuilt: {len(tables['responses'])} responses")


if __name__ == "__main__":
    main()
