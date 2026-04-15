"""Compatibility wrapper for strategy self-tests.

Run:
    PYTHONPATH=. python scripts/test_strategies_selftest.py
"""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET

from scripts.run_strategy_tests_and_export_csv import XML_PATH, run_pytest


def _pytest_is_available() -> bool:
    return importlib.util.find_spec("pytest") is not None


def _read_summary() -> dict[str, int] | None:
    if not XML_PATH.exists():
        return None

    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        return None

    total = int(suite.attrib.get("tests", 0))
    errors = int(suite.attrib.get("errors", 0))
    failures = int(suite.attrib.get("failures", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    failed = failures + errors
    passed = total - failed - skipped

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }


def main() -> int:
    print("Strategy self-test compatibility wrapper", flush=True)

    if not _pytest_is_available():
        print("ERROR: pytest is not installed in this environment.", flush=True)
        print("Install dependencies: pip install -r requirements.txt", flush=True)
        print("Then rerun: PYTHONPATH=. python scripts/test_strategies_selftest.py", flush=True)
        return 2

    rc = run_pytest()

    print("-" * 60)
    summary = _read_summary()
    if summary is None:
        print("Total: 0 | Passed: 0 | Failed: 0 | Skipped: 0")
        print(f"WARNING: Summary unavailable; missing XML report at {XML_PATH}")
    else:
        print(
            "Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}".format(
                **summary,
            )
        )
    print("-" * 60)

    if rc != 0:
        print(f"Self-test failed with pytest exit code {rc}")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
