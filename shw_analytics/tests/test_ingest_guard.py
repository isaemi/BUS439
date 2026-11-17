"""Tests for the loader's guard against emptying the survey tables by accident
(shw_analytics/ingest.py). Uses temporary folders only; never touches data/.

Run:  python tests/test_ingest_guard.py      (or: pytest tests/test_ingest_guard.py)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shw_analytics import ingest  # noqa: E402


def _db_with_survey(folder, responses=3):
    db = Path(folder) / "shw.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE responses (response_id VARCHAR)")
    for i in range(responses):
        con.execute("INSERT INTO responses VALUES (?)", [f"R{i}"])
    con.close()
    return db


def _count(db):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    finally:
        con.close()


def _folders(tmp):
    exports, chr_dir = Path(tmp) / "exports", Path(tmp) / "chr_exports"
    exports.mkdir()
    chr_dir.mkdir()
    return exports, chr_dir


def _raises(kind, fn):
    try:
        fn()
    except kind as e:
        return e
    raise AssertionError(f"expected {kind.__name__}")


def test_an_empty_survey_folder_does_not_remove_existing_survey_data():
    with tempfile.TemporaryDirectory() as tmp:
        exports, chr_dir = _folders(tmp)
        db = _db_with_survey(tmp, responses=3)
        e = _raises(ingest.ExportFormatError, lambda: ingest.load(exports, db, chr_dir))
        assert "3 survey responses" in str(e) and "--chr-only" in str(e)
        assert _count(db) == 3                                   # still there
        assert not db.with_suffix(".building.duckdb").exists() or _count(db) == 3


def test_a_mistyped_folder_does_not_remove_existing_survey_data():
    with tempfile.TemporaryDirectory() as tmp:
        _, chr_dir = _folders(tmp)
        db = _db_with_survey(tmp, responses=2)
        _raises(ingest.ExportFormatError, lambda: ingest.load(Path(tmp) / "exprots", db, chr_dir))
        assert _count(db) == 2


def test_asking_for_appointments_only_gets_past_the_guard():
    with tempfile.TemporaryDirectory() as tmp:
        exports, chr_dir = _folders(tmp)
        db = _db_with_survey(tmp, responses=3)
        # No appointment files either, so it stops at the next check, not at the guard.
        e = _raises(FileNotFoundError, lambda: ingest.load(exports, db, chr_dir, chr_only=True))
        assert "No Qualtrics or CHR" in str(e) and _count(db) == 3


def test_a_first_build_from_appointments_alone_is_allowed():
    with tempfile.TemporaryDirectory() as tmp:
        exports, chr_dir = _folders(tmp)
        db = Path(tmp) / "shw.duckdb"                            # no database yet
        _raises(FileNotFoundError, lambda: ingest.load(exports, db, chr_dir))
        empty = _db_with_survey(tmp, responses=0)                # a database with no survey responses
        _raises(FileNotFoundError, lambda: ingest.load(exports, empty, chr_dir))


def test_appointments_only_build_works_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        exports, chr_dir = _folders(tmp)
        made = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_chr.py"), "--students", "150",
                               "--out", str(chr_dir / "synthetic_chr.csv")], capture_output=True, text=True, cwd=ROOT)
        assert made.returncode == 0, made.stderr[-300:]
        db = Path(tmp) / "shw.duckdb"
        ingest.load(exports, db, chr_dir)                        # first build: allowed
        con = duckdb.connect(str(db), read_only=True)
        try:
            assert con.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
            assert con.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] > 0
        finally:
            con.close()


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
