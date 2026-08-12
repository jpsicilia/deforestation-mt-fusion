"""Check temporal feature extraction from the cleaned S2 series (Phase 1).

    python scripts/test_temporal_features.py

Builds the cleaned dry-season series, extracts temporal features (mean, std,
p5, p95, range, slope) for the vegetation indices, and reports the output band
names + a sanity value. No heavy download (getInfo only).
"""
from __future__ import annotations

import logging

import ee

from deforest_mt.config import load_config
from deforest_mt.gee.auth import init_ee
from deforest_mt.gee.aoi import get_aoi
from deforest_mt.gee.sentinel2 import s2_series
from deforest_mt.gee.temporal_features import stack_temporal_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("test_tf")


def main() -> None:
    cfg = load_config()
    init_ee(cfg["gee"]["project"])

    aoi = get_aoi(cfg["study_area"]["bounds"])
    c = cfg["composite"]
    year = cfg["bitemporal"]["after_year"]
    months = cfg["bitemporal"]["dry_months"]

    # 1. cleaned series
    series = s2_series(
        aoi=aoi, year=year, months=months,
        collection=c["s2_collection"], cloudscore_collection=c["s2_cloudscore"],
        cs_band=c["cs_band"], cs_threshold=c["cs_threshold"],
        bands=c["s2_bands"], indices=c["s2_indices"],
    )
    logger.info("Series scenes: %d", series.size().getInfo())

    # 2. temporal features for the vegetation indices
    feats = stack_temporal_features(series, c["s2_indices"])
    names = feats.bandNames().getInfo()
    logger.info("Temporal feature bands (%d): %s", len(names), names)

    # 3. sanity: NDVI temporal features near the AOI centre
    sample = feats.select(["NDVI_mean", "NDVI_std", "NDVI_range", "NDVI_slope"]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi.centroid(1).buffer(500),
        scale=10, maxPixels=1e6,
    ).getInfo()
    logger.info("NDVI temporal features near centre: %s", sample)
    logger.info("Temporal features OK.")


if __name__ == "__main__":
    main()