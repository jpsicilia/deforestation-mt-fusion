"""Sentinel-2 optical time series (Phase 1).

Builds a CLEANED dry-season time series (not a composite): every scene in the
window, cloud-masked with Cloud Score+, scaled to reflectance, with vegetation
indices added per scene. The series is then summarised by temporal_features.py.

Why a series (not a median composite): a composite collapses the time axis and
blurs a clearing inside the window. Keeping the series lets us extract temporal
features (level, variability, magnitude, direction) that describe the change.
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


def _scale_and_index(img: ee.Image, indices: Sequence[str]) -> ee.Image:
    """Scale one S2 scene to reflectance (0..1) and append vegetation indices."""
    img = img.divide(10000)
    b = {
        "BLUE": img.select("B2"), "GREEN": img.select("B3"),
        "RED": img.select("B4"), "NIR": img.select("B8"),
        "SWIR2": img.select("B12"),
    }
    catalog = {
        "NDVI": img.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        "NBR": img.normalizedDifference(["B8", "B12"]).rename("NBR"),
        "NDWI": img.normalizedDifference(["B3", "B8"]).rename("NDWI"),
        "EVI": img.expression("2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)", b).rename("EVI"),
        "SAVI": img.expression("1.5*(NIR-RED)/(NIR+RED+0.5)", b).rename("SAVI"),
    }
    return img.addBands([catalog[name] for name in indices])


def s2_series(
    aoi: ee.Geometry,
    year: int,
    months: Sequence[int],
    collection: str,
    cloudscore_collection: str,
    cs_band: str,
    cs_threshold: float,
    bands: Sequence[str],
    indices: Sequence[str],
) -> ee.ImageCollection:
    """Cloud-masked dry-season Sentinel-2 time series with indices.

    Args:
        aoi: study-area geometry.
        year: bitemporal endpoint (e.g. 2022 or 2023).
        months: [start_month, end_month] inclusive (dry season).
        collection: S2 SR collection id.
        cloudscore_collection: Cloud Score+ collection id.
        cs_band: clear-confidence band (e.g. 'cs_cdf').
        cs_threshold: keep pixels with clear-confidence >= this.
        bands: reflectance bands to keep.
        indices: vegetation indices to append per scene.

    Returns:
        Cleaned ee.ImageCollection (one image per scene), each carrying the
        reflectance bands + indices, clouds masked out. Ready for
        temporal_features.stack_temporal_features.
    """
    start, end = _dry_range(year, months)
    csp = ee.ImageCollection(cloudscore_collection)

    def _prep(img: ee.Image) -> ee.Image:
        masked = img.updateMask(img.select(cs_band).gte(cs_threshold)).select(bands)
        return _scale_and_index(masked, indices).clip(aoi)

    return (
        ee.ImageCollection(collection)
        .filterBounds(aoi)
        .filterDate(start, end)
        .linkCollection(csp, [cs_band])
        .map(_prep)
    )