from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pipeline import print_report_summary, run_dataset


DATASETS = ("113", "619", "985")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process BlockIQ DAS CSV data with a fixed 4 m gauge."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=("all", *DATASETS),
        default="all",
        help="dataset to process; defaults to all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        print_report_summary(run_dataset(dataset))


if __name__ == "__main__":
    main()
