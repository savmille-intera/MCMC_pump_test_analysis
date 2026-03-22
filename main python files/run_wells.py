#!/usr/bin/env python3
"""
Batch runner for MCMC pump-test analysis.

Reads a list of well directories from a text config file and runs the
full analysis (optimize → minimize → MCMC → predict) for each one.

Usage
-----
  # Single-chain MCMC (default):
  python run_wells.py

  # Gelman-Rubin multi-chain MCMC:
  python run_wells.py --gr

  # Custom wells config file:
  python run_wells.py --wells /path/to/my_wells.txt

Config file format (wells_to_run.txt)
--------------------------------------
  # Lines beginning with # are comments; blank lines are ignored.
  # Paths can be absolute or relative to the repo root.
  ../well CR-15
  ../well CR-16
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MCMC pump-test analysis for a list of well directories."
    )
    parser.add_argument(
        "--gr",
        action="store_true",
        help="Use the Gelman-Rubin multi-chain MCMC variant.",
    )
    parser.add_argument(
        "--wells",
        default=str(Path(__file__).resolve().parent.parent / "wells_to_run.txt"),
        help="Text file listing well directories to process (default: wells_to_run.txt in repo root).",
    )
    return parser.parse_args()


def read_well_dirs(wells_file):
    path = Path(wells_file)
    if not path.exists():
        print(f"Error: wells file not found: {path}")
        sys.exit(1)
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main():
    args = parse_args()

    well_dirs = read_well_dirs(args.wells)

    if not well_dirs:
        print("No well directories listed.")
        sys.exit(1)

    if args.gr:
        from step_test_mcmc_GR import main as run_well
        variant = "Gelman-Rubin (GR)"
    else:
        from step_test_mcmc import main as run_well
        variant = "Single-chain"

    print(f"MCMC variant  : {variant}")
    print(f"Wells to run  : {len(well_dirs)}")
    print(f"Wells file    : {args.wells}\n")

    failed = []
    for well_dir in well_dirs:
        well_path = Path(well_dir)
        if not well_path.exists():
            print(f"[SKIP] Directory not found: {well_path}")
            failed.append(str(well_path))
            continue

        print(f"\n{'=' * 70}")
        print(f"  Well: {well_path.resolve()}")
        print(f"{'=' * 70}")

        t0 = time.perf_counter()
        try:
            run_well(well_dir=str(well_path.resolve()))
        except Exception as exc:
            print(f"[ERROR] {well_path}: {exc}")
            failed.append(str(well_path))
        elapsed = time.perf_counter() - t0
        print(f"  Finished in {elapsed:.1f}s")

    print(f"\n{'=' * 70}")
    print(f"Done. {len(well_dirs) - len(failed)}/{len(well_dirs)} wells completed successfully.")
    if failed:
        print("Failed wells:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()

