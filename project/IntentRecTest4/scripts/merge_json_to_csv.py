#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Any


def flatten(prefix: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}_{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(key, v))
        else:
            out[key] = v
    return out


def collect_metrics(input_dir: Path, include_stats: bool = True) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    # Look for metrics.json in immediate subdirectories (e.g., bert4rec/metrics.json)
    for metrics_path in sorted(input_dir.glob("*/metrics.json")):
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read {metrics_path}: {e}")
            continue

        model = metrics_path.parent.name
        record: Dict[str, Any] = {"model": model, "source": str(metrics_path)}

        # Keep only desired sections
        for section in ("valid", "test"):
            if section in data and isinstance(data[section], dict):
                record.update(flatten(section, data[section]))

        if include_stats and isinstance(data.get("stats"), dict):
            record.update(flatten("stats", data["stats"]))

        records.append(record)

    # Also handle flat files like <model>_metrics.json or metrics.json directly in input_dir
    for metrics_path in sorted(input_dir.glob("*metrics.json")):
        # Avoid double counting nested ones (already handled above)
        if metrics_path.parent != input_dir:
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read {metrics_path}: {e}")
            continue

        model = (
            str(data.get("model"))
            if isinstance(data.get("model"), (str, int))
            else metrics_path.stem.replace("_metrics", "")
        )
        record: Dict[str, Any] = {"model": model, "source": str(metrics_path)}

        for section in ("valid", "test"):
            if section in data and isinstance(data[section], dict):
                record.update(flatten(section, data[section]))

        # Some of these files may include additional top-level fields like loss, epoch
        for extra_key in ("epoch", "loss", "_elapsed_sec"):
            if extra_key in data and not isinstance(data[extra_key], dict):
                record[extra_key] = data[extra_key]

        if include_stats and isinstance(data.get("stats"), dict):
            record.update(flatten("stats", data["stats"]))

        records.append(record)

    return records


def write_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    if not records:
        print("No records found. Nothing to write.")
        return

    # Build stable column order: model first, then valid_*, test_*, stats_* by sorted key
    keys = set().union(*(r.keys() for r in records))
    # Ensure 'model' is first if present
    ordered = [k for k in ["model"] if k in keys]

    def pick(prefix: str) -> List[str]:
        return sorted([k for k in keys if k.startswith(prefix + "_")])

    ordered += pick("valid")
    ordered += pick("test")
    ordered += pick("stats")
    # Any remaining keys not covered (unlikely)
    remaining = sorted([k for k in keys if k not in set(ordered)])
    ordered += remaining

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"Wrote {len(records)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge JSON metrics into a CSV table.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results") / "movielens" / "all",
        help="Directory containing subfolders with metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "movielens" / "all" / "metrics_merged.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Exclude stats_* columns",
    )

    args = parser.parse_args()
    input_dir: Path = args.input_dir
    output_path: Path = args.output
    include_stats: bool = not args.no_stats

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        raise SystemExit(1)

    records = collect_metrics(input_dir, include_stats=include_stats)
    write_csv(records, output_path)


if __name__ == "__main__":
    main()
