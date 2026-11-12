"""Answer labels kept with the data fill gaps; labels in dictionary.yaml win.
Run:  python tests/test_labels.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import config, ingest  # noqa: E402


def _with_data_dir(labels_yaml, fn):
    old = config.DB_PATH
    with tempfile.TemporaryDirectory() as tmp:
        config.DB_PATH = Path(tmp) / "shw.duckdb"
        if labels_yaml is not None:
            (Path(tmp) / "value_labels.yaml").write_text(labels_yaml)
        try:
            return fn()
        finally:
            config.DB_PATH = old


def test_labels_next_to_the_data_are_used():
    d = _with_data_dir("value_labels:\n  D1:\n    '1': Undergraduate\n    '2': Graduate\n", ingest.load_dictionary)
    assert ingest._label(d, "D1", "2") == "Graduate" and ingest._label(d, "D1", 2) == "Graduate"
    assert ingest._label(d, "D1", "9") == "9"                      # an unknown code stays a code


def test_without_that_file_nothing_changes():
    d = _with_data_dir(None, ingest.load_dictionary)
    assert ingest._label(d, "D1", "2") == (d.get("value_labels") or {}).get("D1", {}).get("2", "2")


def test_dictionary_labels_take_priority():
    old = config.DICTIONARY_PATH
    with tempfile.TemporaryDirectory() as tmp:
        real = Path(tmp) / "dictionary.yaml"
        real.write_text("value_labels:\n  D1:\n    '1': Real name\n")
        config.DICTIONARY_PATH = real
        try:
            d = _with_data_dir("value_labels:\n  D1:\n    '1': Synthetic name\n    '2': Graduate\n",
                               ingest.load_dictionary)
        finally:
            config.DICTIONARY_PATH = old
    assert d["value_labels"]["D1"] == {"1": "Real name", "2": "Graduate"}


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1; print("FAIL", name, "->", type(e).__name__, str(e)[:300])
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
