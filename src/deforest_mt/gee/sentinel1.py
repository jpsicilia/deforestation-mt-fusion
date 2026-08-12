"""Sentinel-1 SAR time series (Phase 1).

Builds a CLEANED dry-season SAR series, then temporal_features.py summarises it.

Key differences from optical:
  * No cloud mask -- SAR sees through clouds (its whole point).
  * One orbit only (descending) -- keeps the incidence angle consistent across
    dates, removing one of SAR's noise sources.
  * Speckle filter (focal median) -- reduces the salt-and-pepper radar noise.
  * VV/VH ratio -- bands are in dB (log scale), so the ratio is a subtraction.

GEE's COPERNICUS/S1_GRD is already border-noise/thermal-noise/radiometrically
calibrated and terrain-corrected, delivered as sigma-naught in dB (Ferrari 2023).
"""
from __future__ import annotations

from typing import Sequence

import ee


def _dry_range(year: int, months: Sequence[int]) -> tuple[ee.Date, ee.Date]:
    """[start, end) dates for the inclusive dry-season month range in a year."""
    m0, m1 = months
    start = ee.Date.fromYMD(year, m0, 1)
    end = ee.Date.fromYMD(year, m1, 1).advance(1, "month")
    return start, end


def _prep_scene(img: ee.Image, speckle_radius_m: float) -> ee.Image:
    """Speckle-filter one scene and append the VV/VH ratio (dB subtraction)."""
    # light speckle reduction (focal median); temporal reduction reduces more later
    vv = img.select("VV").focal_median(speckle_radius_m, "circle", "meters").rename("VV")
    vh = img.select("VH").focal_median(speckle_radius_m, "circle", "meters").rename("VH")
    ratio = vv.subtract(vh).rename("VV_VH_ratio")   # dB subtraction = log ratio
    out = ee.Image.cat([vv, vh, ratio])
    # preserve acquisition time so the temporal slope regression can read it
    return out.copyProperties(img, ["system:time_start"])


def s1_series(
    aoi: ee.Geometry,
    year: int,
    months: Sequence[int],
    collection: str,
    orbit: str,
    speckle_radius_m: float = 30.0,
) -> ee.ImageCollection:
    """Cleaned dry-season Sentinel-1 time series with VV, VH and their ratio.

    Args:
        aoi: study-area geometry.
        year: bitemporal endpoint (e.g. 2022 or 2023).
        months: [start_month, end_month] inclusive (dry season).
        collection: S1 GRD collection id.
        orbit: 'DESCENDING' or 'ASCENDING' (keep one for angle consistency).
        speckle_radius_m: focal-median radius for speckle reduction.

    Returns:
        Cleaned ee.ImageCollection (one image per scene), each carrying
        VV, VH (dB) and VV_VH_ratio. Ready for temporal_features.
    """
    start, end = _dry_range(year, months)

    coll = (
        ee.ImageCollection(collection)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    def _prep(img: ee.Image) -> ee.Image:
        return ee.Image(_prep_scene(img, speckle_radius_m)).clip(aoi)

    return coll.map(_prep)