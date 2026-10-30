"""Copy new Qualtrics export files from the SFTP drop folder into data/exports,
then rebuild the database. For when Qualtrics' scheduled export lands on a
different machine from the one running the app.

    python -m shw_analytics.sftp_sync            # sync + ingest
    python -m shw_analytics.sftp_sync --no-ingest

Settings (in .env):
    SFTP_HOST=...      SFTP_PORT=22
    SFTP_USER=...      SFTP_PASSWORD=...     (or SFTP_KEY_FILE=~/.ssh/id_ed25519)
    SFTP_DIR=/path/qualtrics sends files to
    SFTP_PREFIX=shw_survey     (the "File Name Prefix" set in Qualtrics; optional)

Files are only copied, never deleted from the server, and a file already in
data/exports is skipped, so running this twice is harmless.
"""
import argparse
import os
from pathlib import Path

from . import config

EXPORT_DIR = config.DB_PATH.parent / "exports"
SUFFIXES = (".csv", ".zip", ".pgp", ".gpg")


def _connect():
    import paramiko
    host = os.getenv("SFTP_HOST")
    if not host:
        raise SystemExit("Set SFTP_HOST, SFTP_USER and SFTP_PASSWORD (or SFTP_KEY_FILE) in .env")
    transport = paramiko.Transport((host, int(os.getenv("SFTP_PORT", "22"))))
    key_file = os.getenv("SFTP_KEY_FILE")
    if key_file:
        key = paramiko.PKey.from_path(os.path.expanduser(key_file))
        transport.connect(username=os.getenv("SFTP_USER"), pkey=key)
    else:
        transport.connect(username=os.getenv("SFTP_USER"), password=os.getenv("SFTP_PASSWORD"))
    return paramiko.SFTPClient.from_transport(transport)


def _unpack(path):
    """Qualtrics may send a zip; keep the CSV inside. Encrypted files need gpg."""
    import subprocess
    import zipfile
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    (path.parent / f"{path.stem}.csv").write_bytes(z.read(name))
        path.unlink()
    elif path.suffix in (".pgp", ".gpg"):
        out = path.with_suffix("")
        out = out if out.suffix == ".csv" else out.with_suffix(".csv")
        r = subprocess.run(["gpg", "--batch", "--yes", "--output", str(out), "--decrypt", str(path)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Could not decrypt {path.name}: {r.stderr.strip()[-200:]}")
        path.unlink()


def sync(client, remote_dir, local_dir=EXPORT_DIR, prefix=""):
    """Download files that aren't here yet. Returns the names copied."""
    local_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in local_dir.iterdir()}
    copied = []
    for name in sorted(client.listdir(remote_dir)):
        if not name.lower().endswith(SUFFIXES) or (prefix and not name.startswith(prefix)):
            continue
        if Path(name).stem in have or Path(Path(name).stem).stem in have:
            continue
        target = local_dir / name
        client.get(f"{remote_dir.rstrip('/')}/{name}", str(target))
        _unpack(target)
        copied.append(name)
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ingest", action="store_true")
    a = ap.parse_args()
    client = _connect()
    try:
        copied = sync(client, os.getenv("SFTP_DIR", "."), prefix=os.getenv("SFTP_PREFIX", ""))
    finally:
        client.close()
    print(f"Copied {len(copied)} new file(s): {', '.join(copied) or 'none'}")
    if copied and not a.no_ingest:
        from . import ingest
        tables = ingest.load(EXPORT_DIR)
        print(f"Database rebuilt: {len(tables['responses'])} responses")


if __name__ == "__main__":
    main()
