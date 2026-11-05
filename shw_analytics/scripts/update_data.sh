#!/bin/sh
# Scheduled data update: copy new Qualtrics exports from the SFTP drop folder
# (skip this line if Qualtrics drops straight into data/exports on this machine),
# then rebuild the database. Output goes to data/update.log.
#
# cron example, daily at 03:00 (one hour after a 02:00 Qualtrics export):
#   0 3 * * * /path/to/shw_analytics/scripts/update_data.sh
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-.venv/bin/python}"
mkdir -p data
{
  echo "=== $(date) ==="
  if [ -n "$SFTP_HOST" ] || grep -q '^SFTP_HOST=' .env 2>/dev/null; then
    "$PY" -m shw_analytics.sftp_sync --no-ingest
  fi
  "$PY" -m shw_analytics.ingest data/exports | head -3
} >> data/update.log 2>&1
