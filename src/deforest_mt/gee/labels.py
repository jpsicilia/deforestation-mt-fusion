"""MapBiomas transition labels with edge erosion (Phase 1).

Labels come from MapBiomas annual transitions, NOT from DETER (held out as an
independent comparison -> no circularity):
    stable forest (0): forest at t0 AND forest at t1
    deforestation (1): forest at t0 AND anthropic at t1
Everything else is masked.

EDGE EROSION (documented need in Mato Grosso): PRODES/MapBiomas polygon borders
carry small positional uncertainty, and MT clearings are small and amorphous, so
pixels on class boundaries are often mislabelled and degrade training (Cerrado
LSTM study, 2022). We erode each class by `erosion_px` MapBiomas pixels so only
the confident interior ("core") of each region is sampled. Eroding just 1 pixel
removes the ambiguous rim without destroying the small clearings that dominate MT
(MapBiomas labels down to ~1 pixel / 0.09 ha).
"""
from __future__ import annotations

from typing import Sequence

import ee


def build_label_image(
    asset: str,
    year_before: int,
    year_after: int,
    forest_classes: Sequence[int],
    anthropic_classes: Sequence[int],
    stable_value: int = 0,
    deforestation_value: int = 1,
    erosion_px: int = 1,
) -> ee.Image:
    """Binary, edge-eroded label image for the pair (year_before -> year_after).

    Args:
        asset: MapBiomas integration asset id.
        year_before, year_after: bitemporal endpoints.
        forest_classes: MapBiomas codes counted as forest.
        anthropic_classes: codes counted as anthropic (deforestation target).
        stable_value, deforestation_value: label values (0 / 1).
        erosion_px: erode each class by this many MapBiomas (30 m) pixels to drop
            ambiguous boundary pixels. 0 disables erosion.

    Returns:
        ee.Image band 'label' (0/1), masked outside confident cores.
    """
    cls_t0 = ee.Image(asset).select(f"classification_{year_before}")
    cls_t1 = ee.Image(asset).select(f"classification_{year_after}")

    forest_t0 = cls_t0.remap(list(forest_classes), [1] * len(forest_classes), 0)
    forest_t1 = cls_t1.remap(list(forest_classes), [1] * len(forest_classes), 0)
    anth_t1 = cls_t1.remap(list(anthropic_classes), [1] * len(anthropic_classes), 0)

    is_stable = forest_t0.And(forest_t1)     # forest -> forest
    is_defor = forest_t0.And(anth_t1)        # forest -> anthropic

    # erode each class mask to keep only confident interior pixels
    if erosion_px > 0:
        kernel = ee.Kernel.square(radius=erosion_px, units="pixels")
        # a pixel survives only if ALL neighbours share its class (min == 1)
        is_stable = is_stable.reduceNeighborhood(ee.Reducer.min(), kernel).gte(1)
        is_defor = is_defor.reduceNeighborhood(ee.Reducer.min(), kernel).gte(1)

    label = (
        ee.Image.constant(stable_value).updateMask(is_stable)
        .unmask(ee.Image.constant(deforestation_value).updateMask(is_defor),
                sameFootprint=False)
    )
    valid = is_stable.Or(is_defor)
    return label.updateMask(valid).rename("label").toInt8()