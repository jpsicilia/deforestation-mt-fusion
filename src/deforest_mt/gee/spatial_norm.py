"""Hamunyela et al. (2016) spatial normalisation — the project's differentiator.

WHY (answers the drought/false-positive problem):
Climatic effects like drought act at REGIONAL scale (a pixel and its neighbours
fall together), while deforestation acts LOCALLY (only the focal pixel falls).
Dividing a pixel by the median of its healthy neighbours therefore CANCELS the
regional signal (ratio ~ 1) and AMPLIFIES the local one (ratio well below 1).

FORMULA (Hamunyela 2016, Eq. 1):
    sVI = VI_pixel / VI_median
where VI_median is the median of neighbouring pixels whose values are AT OR
ABOVE the 90th percentile in a moving window. The high percentile ensures the
reference is computed from undisturbed (forest) pixels, not from already-cleared
ones -- otherwise the disturbance would be smoothed away. Optimal window at the
Brazilian evergreen site was ~25x25 px.

Shimizu did NOT use this; combining it with an RF fusion model is the novel bit.
Applied to each composite/feature separately, to optical and (verified per data) SAR.
"""
from __future__ import annotations

import ee


def spatial_normalise(
    img: ee.Image,
    band: str,
    window_px: int,
    percentile: int,
) -> ee.Image:
    """Spatially normalise one band (Hamunyela 2016, Eq. 1).

    Args:
        img: image containing `band`.
        band: band to normalise (e.g. 'NDVI', 'VV').
        window_px: moving-window size in pixels (square neighbourhood).
        percentile: reference percentile (e.g. 90) — neighbours at/above this
            define the "healthy" reference.

    Returns:
        Single-band ee.Image '<band>_snorm' = pixel / median(neighbours >= P).
    """
    kernel = ee.Kernel.square(radius=window_px // 2, units="pixels")
    src = img.select(band)

    # 1. the P-th percentile of the neighbourhood (the "healthy forest" threshold)
    p_thresh = src.reduceNeighborhood(
        reducer=ee.Reducer.percentile([percentile]),
        kernel=kernel,
    )

    # 2. median of ONLY the neighbours at/above that percentile
    healthy = src.updateMask(src.gte(p_thresh))
    vi_median = healthy.reduceNeighborhood(
        reducer=ee.Reducer.median(),
        kernel=kernel,
    )

    # 3. divide focal pixel by the healthy-neighbour median
    snorm = src.divide(vi_median).rename(f"{band}_snorm")
    return snorm