"""Area of interest (AOI) helpers.

The AOI is a rectangle defined once in config.yaml as [W, S, E, N]. Everything
downstream clips to this geometry, so the study area lives in exactly one place.
"""
from __future__ import annotations

from typing import Sequence

import ee


def get_aoi(bounds: Sequence[float]) -> ee.Geometry:
    """Build an EE rectangle from [West, South, East, North] degrees.

    Args:
        bounds: [W, S, E, N] in EPSG:4326.

    Returns:
        ee.Geometry.Rectangle covering the study area.
    """
    west, south, east, north = bounds
    return ee.Geometry.Rectangle(
        [west, south, east, north], proj="EPSG:4326", geodesic=False
    )


def aoi_area_km2(aoi: ee.Geometry) -> float:
    """Return the AOI area in km² (server call)."""
    return aoi.area(maxError=1).divide(1e6).getInfo()
