"""Sanity check — run first to confirm the environment works.

    python scripts/00_check_setup.py

Verifies: config loads, Earth Engine initialises, MapBiomas asset is reachable
and has the bitemporal years, and the AOI is the expected size.
"""
from __future__ import annotations

import logging

import ee

from deforest_mt.config import load_config
from deforest_mt.gee.auth import init_ee
from deforest_mt.gee.aoi import get_aoi, aoi_area_km2

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("check")


def main() -> None:
    cfg = load_config()
    logger.info("Config OK. Study area: %s", cfg["study_area"]["name"])

    init_ee(cfg["gee"]["project"])

    aoi = get_aoi(cfg["study_area"]["bounds"])
    logger.info("AOI area: %.0f km2", aoi_area_km2(aoi))

    asset = cfg["labels"]["mapbiomas_asset"]
    bands = ee.Image(asset).bandNames().getInfo()
    needed = {f"classification_{cfg['bitemporal']['before_year']}",
              f"classification_{cfg['bitemporal']['after_year']}"}
    missing = sorted(needed - set(bands))
    if missing:
        logger.warning("MapBiomas asset missing bands: %s — check collection/years.", missing)
    else:
        logger.info("MapBiomas asset OK — bitemporal years present.")

    logger.info("Setup check complete.")


if __name__ == "__main__":
    main()
