"""Records every request this app sends to the LTIC gateway, so the team can see its own share
of the shared API key. Nothing about the content of a request is recorded: no questions, no
answers, no comments. Only the date, what kind of request it was, the model, token counts and
the cost the gateway reports.

    python -m shw_analytics.usage                # totals: all time, this month, by kind and model
    python -m shw_analytics.usage --days 7       # just the last week

The log is data/usage_log.csv, which git ignores (the data/ folder never goes in the repo).
"""
import argparse
import csv
import datetime as dt
import threading
from pathlib import Path

from . import config

LOG = config.DB_PATH.parent / "usage_log.csv"
FIELDS = ["time", "kind", "model", "input_tokens", "output_tokens", "cost", "status"]
_lock = threading.Lock()


def _write(row):
    with _lock:
        new = not LOG.exists()
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def _record(kind, parsed, headers):
    """Log one line for a completed request. Never raises, so logging can never break an answer."""
    try:
        use = getattr(parsed, "usage", None)
        cost = headers.get("x-litellm-response-cost") if headers is not None else None
        _write({"time": dt.datetime.now().isoformat(timespec="seconds"), "kind": kind,
                "model": getattr(parsed, "model", "") or "",
                "input_tokens": getattr(use, "prompt_tokens", "") if use else "",
                "output_tokens": (getattr(use, "completion_tokens", "") or "") if use else "",
                "cost": cost if cost not in (None, "None") else "", "status": 200})
    except Exception:
        pass


def attach(client):
    """Make `client` log every chat and embedding request. Uses the client's own raw-response
    calls, so it needs no extra library. If anything about the client is unexpected, it is
    left exactly as it was."""
    def wrap(resource, kind):
        raw_create = resource.with_raw_response.create

        def create(*args, **kwargs):
            raw = raw_create(*args, **kwargs)
            parsed = raw.parse()
            _record(kind, parsed, getattr(raw, "headers", None))
            return parsed
        resource.create = create
    try:
        wrap(client.chat.completions, "chat")
        wrap(client.embeddings, "embedding")
    except Exception:
        pass
    return client


def summary(days=None):
    import pandas as pd
    if not LOG.exists():
        return "No gateway requests recorded yet. The log starts with the next question you ask."
    df = pd.read_csv(LOG, parse_dates=["time"])
    if days:
        df = df[df["time"] >= pd.Timestamp.now() - pd.Timedelta(days=days)]
    if df.empty:
        return "No gateway requests in that period."
    for c in ("input_tokens", "output_tokens", "cost"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    priced = df["cost"].notna().mean()

    def line(d, label):
        total = d["cost"].sum()
        cost = (f"${total:,.4f}" if total >= 0.01 else f"${total:.6f}") if d["cost"].notna().any() else "cost not reported"
        return (f"{label:<26}{len(d):>7,} requests {int(d['input_tokens'].sum()):>12,} in "
                f"{int(d['output_tokens'].fillna(0).sum()):>10,} out   {cost}")

    out = [f"Gateway usage recorded by this app since {df['time'].min():%d %b %Y}"
           + (f" (last {days} days)" if days else ""), "",
           line(df, "Total")]
    month = df[df["time"] >= pd.Timestamp.now().normalize().replace(day=1)]
    if len(month) and len(month) < len(df):
        out.append(line(month, "This month"))
    out += ["", "By kind:"] + [line(d, f"  {k}") for k, d in df.groupby("kind")]
    out += ["", "By model:"] + [line(d, f"  {m}") for m, d in df.groupby("model")]
    errors = int((df["status"] >= 400).sum())
    out += ["", f"Failed requests: {errors:,}" if errors else "No failed requests."]
    if priced < 1:
        out.append("Some requests had no cost from the gateway; their tokens are counted, their cost is not.")
    out.append("This covers this app on this computer only. The shared key's total includes the other teams.")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int)
    print(summary(ap.parse_args().days))
