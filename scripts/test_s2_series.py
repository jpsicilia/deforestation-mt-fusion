"""Quick check for the Sentinel-2 cleaned time series (Phase 1).

    python scripts/test_s2_composite.py

Verifies the cleaned series builds over the AOI: how many scenes it has, which
bands each scene carries, and a sanity NDVI value. No heavy download (getInfo).
"""
from __future__ import annotations

import logging

import ee

from deforest_mt.config import load_config
from deforest_mt.gee.auth import init_ee
from deforest_mt.gee.aoi import get_aoi
from deforest_mt.gee.sentinel2 import s2_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("test_s2")


def main() -> None:
    cfg = load_config()
    init_ee(cfg["gee"]["project"])

    aoi = get_aoi(cfg["study_area"]["bounds"])
    c = cfg["composite"]
    year = cfg["bitemporal"]["after_year"]
    months = cfg["bitemporal"]["dry_months"]

    series = s2_series(
        aoi=aoi, year=year, months=months,
        collection=c["s2_collection"], cloudscore_collection=c["s2_cloudscore"],
        cs_band=c["cs_band"], cs_threshold=c["cs_threshold"],
        bands=c["s2_bands"], indices=c["s2_indices"],
    )

    n = series.size().getInfo()
    logger.info("Sentinel-2 scenes in %d dry-season series: %d", year, n)

    first_bands = ee.Image(series.first()).bandNames().getInfo()
    logger.info("Per-scene bands: %s", first_bands)

    # sanity: mean NDVI across the whole series near the AOI centre
    mean_ndvi = (
        series.select("NDVI").mean()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi.centroid(1).buffer(500),
            scale=10, maxPixels=1e6,
        ).getInfo()
    )
    logger.info("Series mean NDVI near AOI centre: %s", mean_ndvi)
    logger.info("S2 series OK.")


if __name__ == "__main__":
    main()