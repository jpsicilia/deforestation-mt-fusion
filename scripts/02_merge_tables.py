"""Merge the per-batch feature CSVs into one training table (Phase 1 -> Phase 2).

    python scripts/02_merge_tables.py

Reads every features_*_b*.csv in data/raw/, concatenates them, drops GEE's
bookkeeping columns, reports class balance, and writes one clean table.
"""
from __future__ import annotations

import glob
import logging

import pandas as pd

from deforest_mt.config import DATA_RAW, load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("merge")


def main() -> None:
    cfg = load_config()
    bt = cfg["bitemporal"]
    pattern = str(DATA_RAW / f"features_{bt['before_year']}_{bt['after_year']}_b*.csv")

    files = sorted(glob.glob(pattern))
    if not files:
        logger.error("No batch CSVs found matching %s", pattern)
        logger.error("Download the features_*_b*.csv from Drive into data/raw/ first.")
        return
    logger.info("Merging %d batch files", len(files))

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    # drop GEE bookkeeping columns if present
    for col in ["system:index", ".geo"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # de-duplicate in case of overlap, report balance
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Rows: %d (%d after de-dup)", before, len(df))
    logger.info("Class balance:\n%s", df["label"].value_counts().to_string())
    logger.info("Feature columns: %d", df.shape[1] - 1)

    out = DATA_RAW / f"training_table_{bt['before_year']}_{bt['after_year']}.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote merged table -> %s", out)


if __name__ == "__main__":
    main()