"""Check the Sentinel-1 SAR series + temporal features (Phase 1).

    python scripts/test_s1_series.py
"""
from __future__ import annotations

import logging

import ee

from deforest_mt.config import load_config
from deforest_mt.gee.auth import init_ee
from deforest_mt.gee.aoi import get_aoi
from deforest_mt.gee.sentinel1 import s1_series
from deforest_mt.gee.temporal_features import stack_temporal_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("test_s1")


def main() -> None:
    cfg = load_config()
    init_ee(cfg["gee"]["project"])

    aoi = get_aoi(cfg["study_area"]["bounds"])
    c = cfg["composite"]
    year = cfg["bitemporal"]["after_year"]
    months = cfg["bitemporal"]["dry_months"]

    series = s1_series(
        aoi=aoi, year=year, months=months,
        collection=c["s1_collection"], orbit=c["s1_orbit"],
    )
    logger.info("Sentinel-1 scenes in %d dry season: %d", year, series.size().getInfo())
    logger.info("Per-scene bands: %s", ee.Image(series.first()).bandNames().getInfo())

    feats = stack_temporal_features(series, ["VV", "VH", "VV_VH_ratio"])
    names = feats.bandNames().getInfo()
    logger.info("SAR temporal feature bands (%d): %s", len(names), names)

    sample = feats.select(["VV_mean", "VV_std", "VH_mean"]).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi.centroid(1).buffer(500),
        scale=10, maxPixels=1e6,
    ).getInfo()
    logger.info("SAR temporal features near centre: %s", sample)
    logger.info("S1 series OK.")


if __name__ == "__main__":
    main()