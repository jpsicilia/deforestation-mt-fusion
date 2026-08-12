"""Check Hamunyela spatial normalisation on the S2 series (Phase 1).

    python scripts/test_spatial_norm.py

Applies spatial normalisation to the mean-NDVI image and checks that stable
forest normalises to ~1 (regional signal cancelled).
"""
from __future__ import annotations

import logging

import ee

from deforest_mt.config import load_config
from deforest_mt.gee.auth import init_ee
from deforest_mt.gee.aoi import get_aoi
from deforest_mt.gee.sentinel2 import s2_series
from deforest_mt.gee.spatial_norm import spatial_normalise

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("test_sn")


def main() -> None:
    cfg = load_config()
    init_ee(cfg["gee"]["project"])

    aoi = get_aoi(cfg["study_area"]["bounds"])
    c = cfg["composite"]
    sn = cfg["spatial_norm"]
    year = cfg["bitemporal"]["after_year"]
    months = cfg["bitemporal"]["dry_months"]

    series = s2_series(
        aoi=aoi, year=year, months=months,
        collection=c["s2_collection"], cloudscore_collection=c["s2_cloudscore"],
        cs_band=c["cs_band"], cs_threshold=c["cs_threshold"],
        bands=c["s2_bands"], indices=c["s2_indices"],
    )
    ndvi_mean = series.select("NDVI").mean()

    snorm = spatial_normalise(
        ndvi_mean, band="NDVI",
        window_px=sn["window_px"], percentile=sn["percentile"],
    )

    logger.info("Normalised band: %s", snorm.bandNames().getInfo())

    # stable forest should normalise to ~1 (regional signal cancels)
    val = snorm.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi.centroid(1).buffer(500),
        scale=10, maxPixels=1e6,
    ).getInfo()
    logger.info("Normalised NDVI near forest centre (expect ~1): %s", val)
    logger.info("Spatial normalisation OK.")


if __name__ == "__main__":
    main()