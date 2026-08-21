"""Read-only real-pilot coverage and Demo 2 evidence report."""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

from pilot_reporting import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect real pilot coverage without mixing synthetic data.")
    parser.add_argument("--demo2", action="store_true", help="Emit the Demo 2 evidence-focused report")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--start", help="Collection Day 1 (YYYY-MM-DD); defaults to first real event")
    args = parser.parse_args()
    start = date.fromisoformat(args.start) if args.start else None
    report = build_report(os.getenv("CHRONIS_DB_PATH", "events.db"), start=start)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    classification = report["classification"]
    print("=== CHRONIS REAL PILOT INSPECTION ===")
    print(f"Real pilot events: {classification['real_pilot_events']}")
    print(f"Synthetic events excluded: {classification['synthetic_events']}")
    print(f"Unattributed legacy events excluded: {classification['unattributed_legacy_events']}")
    print(f"Coverage start: {report['coverage_start'] or 'not established'}")
    for pilot_id, item in report["coverage"].items():
        print(f"{pilot_id}: {item['status']} | missing days: {item['missing_days']}")
    print(f"Malformed records: {len(report['malformed_events'])}; duplicates: {len(report['duplicates'])}")
    if args.demo2:
        print("=== DEMO 2 EVIDENCE ===")
        print(report["demo2_status"])
        for candidate in report["candidate_patterns"]:
            print(json.dumps(candidate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
