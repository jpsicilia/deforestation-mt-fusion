"""Feature assembly (Phase 1 closer).

Builds the full bitemporal predictor stack and samples it at the labelled points
to produce the training table for the Random Forest.

For each year (before, after):
    S2 series -> temporal features (+ spatially-normalised versions)
    S1 series -> temporal features (+ spatially-normalised versions)
Then:
    diff = after - before   (the bitemporal change, with direction/sign)
    stack = [before features, after features, diffs] + label
    stratified sample -> table -> export

The three scenarios (S2 / S1 / fusion) and with/without spatial normalisation are
obtained downstream by selecting column subsets of this one table.
"""
from __future__ import annotations

from typing import Sequence

import ee

from deforest_mt.gee.sentinel2 import s2_series
from deforest_mt.gee.sentinel1 import s1_series
from deforest_mt.gee.temporal_features import stack_temporal_features
from deforest_mt.gee.spatial_norm import spatial_normalise


def _year_stack(aoi: ee.Geometry, year: int, cfg: dict) -> ee.Image:
    """Full temporal-feature stack (S2 + S1, raw + spatially normalised) for one year."""
    c = cfg["composite"]
    sn = cfg["spatial_norm"]
    months = cfg["bitemporal"]["dry_months"]

    # --- optical ---
    s2 = s2_series(
        aoi=aoi, year=year, months=months,
        collection=c["s2_collection"], cloudscore_collection=c["s2_cloudscore"],
        cs_band=c["cs_band"], cs_threshold=c["cs_threshold"],
        bands=c["s2_bands"], indices=c["s2_indices"],
    )
    s2_feats = stack_temporal_features(s2, c["s2_indices"])

    # --- SAR ---
    s1 = s1_series(
        aoi=aoi, year=year, months=months,
        collection=c["s1_collection"], orbit=c["s1_orbit"],
    )
    s1_bands = ["VV", "VH", "VV_VH_ratio"]
    s1_feats = stack_temporal_features(s1, s1_bands)

    stack = s2_feats.addBands(s1_feats)

    # --- Hamunyela spatial normalisation on the mean images (the differentiator) ---
    if sn["enabled"]:
        norm_imgs = []
        for band in sn["optical_bands"]:
            norm_imgs.append(spatial_normalise(
                s2.select(band).mean(), band, sn["window_px"], sn["percentile"]))
        for band in sn["sar_bands"]:
            norm_imgs.append(spatial_normalise(
                s1.select(band).mean(), band, sn["window_px"], sn["percentile"]))
        stack = stack.addBands(ee.Image.cat(norm_imgs))

    # tag every band with the year so before/after don't collide when stacked
    old = stack.bandNames()
    new = old.map(lambda b: ee.String(b).cat(f"_y{year}"))
    return stack.rename(new)


def build_bitemporal_stack(aoi: ee.Geometry, cfg: dict) -> ee.Image:
    """Assemble [before, after, differences] into one predictor image.

    Differences (after - before) carry the DIRECTION of change: a strong
    negative diff in NDVI features = deforestation; positive = regrowth.
    """
    y0 = cfg["bitemporal"]["before_year"]
    y1 = cfg["bitemporal"]["after_year"]

    before = _year_stack(aoi, y0, cfg)
    after = _year_stack(aoi, y1, cfg)

    # differences: strip the year suffix and subtract matching features
    def _strip(names: ee.List, suffix: str) -> ee.List:
        return names.map(lambda b: ee.String(b).replace(suffix + "$", ""))

    before_r = before.rename(_strip(before.bandNames(), f"_y{y0}"))
    after_r = after.rename(_strip(after.bandNames(), f"_y{y1}"))

    common = before_r.bandNames().filter(
        ee.Filter.inList("item", after_r.bandNames()))
    diff_names = common.map(lambda b: ee.String(b).cat("_diff"))
    diff = after_r.select(common).subtract(before_r.select(common)).rename(diff_names)

    return before.addBands(after).addBands(diff)


def sample_training_table(
    stack: ee.Image,
    label_image: ee.Image,
    aoi: ee.Geometry,
    n_per_class: int,
    scale: int,
    seed: int,
) -> ee.FeatureCollection:
    """Stratified-sample label + all features in one pass (no fragile join)."""
    combined = label_image.addBands(stack)
    return combined.stratifiedSample(
        numPoints=n_per_class,
        classBand="label",
        region=aoi,
        scale=scale,
        seed=seed,
        geometries=True,
        dropNulls=True,
        tileScale=4,
    )