"""Run strategy pytest suite and export CSV results in one command.

Usage:
    PYTHONPATH=. python3 scripts/run_strategy_tests_and_export_csv.py
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "test_reports"
XML_PATH = REPORT_DIR / "strategies_pytest_results.xml"
CSV_PATH = REPORT_DIR / "strategies_pytest_results.csv"
HISTORY_DIR = REPORT_DIR / "history"


def run_pytest() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/strategies",
        "-q",
        f"--junitxml={XML_PATH}",
    ]
    print("Running:", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    return int(result.returncode)


def convert_xml_to_csv() -> str:
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        raise RuntimeError("No testsuite found in pytest XML output")

    summary = {
        "tests": int(suite.attrib.get("tests", 0)),
        "errors": int(suite.attrib.get("errors", 0)),
        "failures": int(suite.attrib.get("failures", 0)),
        "skipped": int(suite.attrib.get("skipped", 0)),
        "time": float(suite.attrib.get("time", 0.0)),
    }
    summary["passed"] = (
        summary["tests"]
        - summary["errors"]
        - summary["failures"]
        - summary["skipped"]
    )

    run_dt = datetime.now(timezone.utc).replace(microsecond=0)
    run_ts = run_dt.isoformat().replace("+00:00", "Z")

    rows = []
    for case in suite.findall("testcase"):
        status = "passed"
        message = ""
        detail = ""

        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")

        if failure is not None:
            status = "failed"
            message = failure.attrib.get("message", "")
            detail = (failure.text or "").strip()
        elif error is not None:
            status = "error"
            message = error.attrib.get("message", "")
            detail = (error.text or "").strip()
        elif skipped is not None:
            status = "skipped"
            message = skipped.attrib.get("message", "")
            detail = (skipped.text or "").strip()

        rows.append(
            {
                "run_timestamp_utc": run_ts,
                "suite_name": suite.attrib.get("name", ""),
                "test_class": case.attrib.get("classname", ""),
                "test_name": case.attrib.get("name", ""),
                "status": status,
                "duration_sec": case.attrib.get("time", ""),
                "message": message,
                "detail": detail,
                "total_tests": summary["tests"],
                "passed": summary["passed"],
                "failed": summary["failures"],
                "errors": summary["errors"],
                "skipped": summary["skipped"],
                "total_duration_sec": summary["time"],
            }
        )

    fieldnames = [
        "run_timestamp_utc",
        "suite_name",
        "test_class",
        "test_name",
        "status",
        "duration_sec",
        "message",
        "detail",
        "total_tests",
        "passed",
        "failed",
        "errors",
        "skipped",
        "total_duration_sec",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return run_dt.strftime("%Y%m%dT%H%M%SZ")


def save_history_copies(run_stamp: str) -> tuple[Path, Path]:
    xml_hist = HISTORY_DIR / f"strategies_pytest_results_{run_stamp}.xml"
    csv_hist = HISTORY_DIR / f"strategies_pytest_results_{run_stamp}.csv"
    shutil.copyfile(XML_PATH, xml_hist)
    shutil.copyfile(CSV_PATH, csv_hist)
    return xml_hist, csv_hist


def main() -> int:
    rc = run_pytest()

    if not XML_PATH.exists():
        print("ERROR: pytest XML report was not generated.")
        return rc if rc != 0 else 2

    run_stamp = convert_xml_to_csv()
    xml_hist, csv_hist = save_history_copies(run_stamp)
    print(f"XML report: {XML_PATH}")
    print(f"CSV report: {CSV_PATH}")
    print(f"History XML: {xml_hist}")
    print(f"History CSV: {csv_hist}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
