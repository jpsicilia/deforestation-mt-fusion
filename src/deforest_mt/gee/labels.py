"""MapBiomas transition labels (Phase 1 / Phase 0 logic).

Labels come from MapBiomas annual transitions, NOT from DETER (which is held out
as an independent comparison -> no circularity):
    stable forest (0): forest at t0 AND forest at t1
    deforestation (1): forest at t0 AND anthropic at t1
Everything else is masked (ambiguous pixels must not pollute training).
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
) -> ee.Image:
    """Binary label image for the bitemporal pair (year_before -> year_after).

    Returns an ee.Image band 'label' (0/1), masked elsewhere.
    """
    cls_t0 = ee.Image(asset).select(f"classification_{year_before}")
    cls_t1 = ee.Image(asset).select(f"classification_{year_after}")

    forest_t0 = cls_t0.remap(list(forest_classes), [1] * len(forest_classes), 0)
    forest_t1 = cls_t1.remap(list(forest_classes), [1] * len(forest_classes), 0)
    anth_t1 = cls_t1.remap(list(anthropic_classes), [1] * len(anthropic_classes), 0)

    is_stable = forest_t0.And(forest_t1)     # forest -> forest
    is_defor = forest_t0.And(anth_t1)        # forest -> anthropic

    label = (
        ee.Image.constant(stable_value).updateMask(is_stable)
        .unmask(ee.Image.constant(deforestation_value).updateMask(is_defor),
                sameFootprint=False)
    )
    valid = is_stable.Or(is_defor)
    return label.updateMask(valid).rename("label").toInt8()