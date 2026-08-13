"""Phase 1 closer — build the bitemporal feature stack and export the table.

    python scripts/01_build_features.py

Pipeline:
    1. build the binary label image (MapBiomas transitions, before -> after)
    2. build the bitemporal predictor stack (S2 + S1 temporal features, raw +
       spatially normalised, for both years, plus their differences)
    3. stratified-sample label + features at n_per_class points per class
    4. export the table to Google Drive as CSV

The exported CSV (one row per point: label, all features, lon, lat) is the
hand-off to the local Random Forest (Phase 2).
"""
from __future__ import annotations

import logging

from deforest_mt.config import load_config
from deforest_mt.gee.auth import init_ee
from deforest_mt.gee.aoi import get_aoi, aoi_area_km2
from deforest_mt.gee.labels import build_label_image
from deforest_mt.gee.features import build_bitemporal_stack, sample_training_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("phase1")


def main() -> None:
    cfg = load_config()
    init_ee(cfg["gee"]["project"])

    aoi = get_aoi(cfg["study_area"]["bounds"])
    logger.info("AOI area: %.0f km2", aoi_area_km2(aoi))

    lab = cfg["labels"]
    bt = cfg["bitemporal"]
    smp = cfg["sampling"]

    # 1. labels
    label_img = build_label_image(
        asset=lab["mapbiomas_asset"],
        year_before=bt["before_year"],
        year_after=bt["after_year"],
        forest_classes=lab["forest_classes"],
        anthropic_classes=lab["anthropic_classes"],
        stable_value=lab["class_values"]["stable_forest"],
        deforestation_value=lab["class_values"]["deforestation"],
    ).clip(aoi)
    logger.info("Label image built (%d -> %d)", bt["before_year"], bt["after_year"])

    # 2. bitemporal predictor stack
    stack = build_bitemporal_stack(aoi, cfg)
    n_bands = stack.bandNames().size().getInfo()
    logger.info("Bitemporal stack: %d feature bands", n_bands)

    # 3. sample
    samples = sample_training_table(
        stack=stack, label_image=label_img, aoi=aoi,
        n_per_class=smp["n_per_class"], scale=smp["feature_scale"], seed=smp["seed"],
    )

    # 4. export
    task = __import__("ee").batch.Export.table.toDrive(
        collection=samples,
        description=f"features_{bt['before_year']}_{bt['after_year']}",
        folder=cfg["gee"]["export_folder"],
        fileFormat="CSV",
    )
    task.start()
    logger.info("Export started -> Drive/%s (features_%d_%d.csv)",
                cfg["gee"]["export_folder"], bt["before_year"], bt["after_year"])
    logger.info("Monitor at https://code.earthengine.google.com/tasks")
    logger.info("When done, download the CSV into data/raw/ for Phase 2.")


if __name__ == "__main__":
    main()