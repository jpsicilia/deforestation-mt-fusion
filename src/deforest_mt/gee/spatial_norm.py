"""Hamunyela et al. (2016) spatial normalisation — the project's differentiator.

sVI = VI_pixel / median(neighbours >= P90) in a moving window. High percentile =
reference is undisturbed forest, so regional drought cancels (~1) and local
deforestation drops (<1). Window ~25x25 at the Brazilian evergreen site.

Two fixes over the first version:
1. skipMasked=False on the neighbourhood median, so the median of healthy
   neighbours is assigned to the focal pixel even when the focal pixel is below
   its own P90 (i.e. masked). Without this, most pixels returned NaN.
2. SAR bands arrive in dB (logarithmic). Hamunyela's ratio assumes a linear
   quantity, so for SAR we convert dB -> linear power (10^(dB/10)) BEFORE
   normalising, then take the ratio. Optical indices (already linear/positive)
   are normalised directly.
"""
from __future__ import annotations

import ee

# bands that arrive in dB and must be linearised before the ratio
_SAR_BANDS = {"VV", "VH", "VV_VH_ratio"}


def spatial_normalise(
    img: ee.Image,
    band: str,
    window_px: int,
    percentile: int,
) -> ee.Image:
    """Spatially normalise one band (Hamunyela 2016, Eq. 1).

    Optical: ratio on the value directly. SAR (dB): dB->linear, then ratio.
    Returns single-band ee.Image '<band>_snorm'.
    """
    kernel = ee.Kernel.square(radius=window_px // 2, units="pixels")
    src = img.select(band)

    # SAR is in dB (logarithmic): convert to linear power before the ratio
    if band in _SAR_BANDS:
        src = ee.Image(10.0).pow(src.divide(10.0)).rename(band)

    # 1. neighbourhood P-th percentile (healthy-forest threshold); align band name
    p_thresh = src.reduceNeighborhood(
        reducer=ee.Reducer.percentile([percentile]),
        kernel=kernel,
    ).rename(band)

    # 2. median of neighbours >= threshold; skipMasked=False so the focal pixel
    #    receives the healthy-neighbour median even when it is itself masked
    healthy = src.updateMask(src.gte(p_thresh))
    vi_median = healthy.reduceNeighborhood(
        reducer=ee.Reducer.median(),
        kernel=kernel,
        skipMasked=False,
    ).rename(band)

    # 3. ratio: focal value / healthy-neighbour median (adimensional)
    snorm = src.divide(vi_median).rename(f"{band}_snorm")
    return snorm
