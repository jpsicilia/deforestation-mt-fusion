"""Temporal feature extraction from a cleaned image time series (Phase 1 core).

WHY (not a composite): a median composite collapses the time axis into one
number and *blurs* a clearing that happens inside the window. Instead we
describe each pixel's temporal *behaviour* with several features, so the Random
Forest can tell "changed downward" (deforestation) from "stable" (forest) and
from "changed upward" (regrowth). Sensor-agnostic: same reduction for S2 and S1.

FEATURE SET — every feature is backed by primary-source literature:
  * mean, std ............... INPE (Grings/Carvalho 2019): annual statistics
    (mean, standard deviation, amplitude) over a Sentinel-1 series for ML
    deforestation mapping in GEE.
  * range as P95 - P5 ....... Hirschmugl/Deutscher 2020: the 5th-95th percentile
    temporal range is more robust to outliers than raw min/max.
  * slope (SIGNED trend) .... Verbesselt/BFAST 2012 & Hirschmugl 2020: a feature
    that considers temporal *direction* separates deforestation (down) from
    regrowth (up). This answers the regrowth problem.

The final feature set is NOT decided here: this is the literature-backed
candidate. Permutation importance + ablation on OUR data decide what stays
(Dobrinic 2021 kept ~1/4 of variables this way).
"""
from __future__ import annotations

from typing import Sequence

import ee


def _add_time_band(img: ee.Image) -> ee.Image:
    """Attach a fractional-year time band 't' used for the slope regression."""
    t = ee.Image(img.date().difference(ee.Date("1970-01-01"), "year")).toFloat()
    return img.addBands(t.rename("t"))


def temporal_features(series: ee.ImageCollection, band: str) -> ee.Image:
    """Reduce a cleaned single-band series to per-pixel temporal features.

    Args:
        series: cleaned ImageCollection (clouds/speckle already handled),
            each image carrying `band`.
        band: variable to summarise (e.g. 'NDVI', 'VV').

    Returns:
        Multi-band ee.Image: <band>_mean, <band>_std, <band>_p5, <band>_p95,
        <band>_range (p95-p5), <band>_slope (signed trend).
    """
    s = series.select(band)

    mean = s.mean().rename(f"{band}_mean")
    std = s.reduce(ee.Reducer.stdDev()).rename(f"{band}_std")

    # robust range: 5th-95th percentile (Hirschmugl 2020) instead of raw min/max
    pcts = s.reduce(ee.Reducer.percentile([5, 95]))
    p5 = pcts.select(0).rename(f"{band}_p5")
    p95 = pcts.select(1).rename(f"{band}_p95")
    rng = p95.subtract(p5).rename(f"{band}_range")

    # signed temporal slope via linear fit value ~ time (direction of change)
    witht = series.map(_add_time_band).select(["t", band])
    slope = witht.reduce(ee.Reducer.linearFit()).select("scale").rename(f"{band}_slope")

    return ee.Image.cat([mean, std, p5, p95, rng, slope])


def stack_temporal_features(
    series: ee.ImageCollection, bands: Sequence[str]
) -> ee.Image:
    """Apply temporal_features to several bands and stack the result.

    Args:
        series: cleaned multi-band series.
        bands: bands to summarise (e.g. ['NDVI','NBR','NDWI'] or ['VV','VH']).

    Returns:
        One ee.Image with temporal features for every requested band.
    """
    return ee.Image.cat([temporal_features(series, b) for b in bands])