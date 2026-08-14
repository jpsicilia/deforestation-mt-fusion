"""Count available labelled pixels per class (before exporting points).

    python scripts/00b_count_labels.py

Tells you the ceiling: how many deforestation vs stable-forest pixels exist in
the AOI. The rarer class (deforestation) caps how many balanced points you can
sample. Use this to choose n_per_class generously but realistically.
"""
from __future__ import annotations

import logging

import ee

from deforest_mt.config import load_config
from deforest_mt.gee.auth import init_ee
from deforest_mt.gee.aoi import get_aoi
from deforest_mt.gee.labels import build_label_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("count")


def main() -> None:
    cfg = load_config()
    init_ee(cfg["gee"]["project"])

    aoi = get_aoi(cfg["study_area"]["bounds"])
    lab = cfg["labels"]
    bt = cfg["bitemporal"]

    label_img = build_label_image(
        asset=lab["mapbiomas_asset"],
        year_before=bt["before_year"],
        year_after=bt["after_year"],
        forest_classes=lab["forest_classes"],
        anthropic_classes=lab["anthropic_classes"],
        stable_value=lab["class_values"]["stable_forest"],
        deforestation_value=lab["class_values"]["deforestation"],
    ).clip(aoi)

    # count pixels per label at 30 m (MapBiomas native)
    counts = label_img.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=aoi,
        scale=30,
        maxPixels=1e10,
        tileScale=8,
    ).getInfo()

    hist = counts.get("label", {})
    stable = int(hist.get("0", 0))
    defor = int(hist.get("1", 0))
    logger.info("Stable forest pixels (label 0): %d", stable)
    logger.info("Deforestation pixels (label 1): %d", defor)
    logger.info("Rarer class (caps balanced sampling): %d", min(stable, defor))
    logger.info("A safe generous n_per_class would be ~%d", int(min(stable, defor) * 0.6))


if __name__ == "__main__":
    main()