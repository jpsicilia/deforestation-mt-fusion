"""Phase 1 closer — build the bitemporal feature stack and export in BATCHES.

    python scripts/01_build_features.py

The full stack (159 features) x thousands of points times the Hamunyela spatial
normalisation is too heavy for one GEE export (times out). We split the sampling
into N batches (by a random batch id) that each complete, then merge the CSVs
locally with 02_merge_tables.py. No data is lost -- it's the same points, split.
"""
from __future__ import annotations

import logging

import ee

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

    lab, bt, smp = cfg["labels"], cfg["bitemporal"], cfg["sampling"]
    n_batches = smp.get("n_batches", 4)
    per_batch = smp["n_per_class"] // n_batches

    # labels (edge-eroded)
    label_img = build_label_image(
        asset=lab["mapbiomas_asset"],
        year_before=bt["before_year"], year_after=bt["after_year"],
        forest_classes=lab["forest_classes"], anthropic_classes=lab["anthropic_classes"],
        stable_value=lab["class_values"]["stable_forest"],
        deforestation_value=lab["class_values"]["deforestation"],
        erosion_px=lab.get("erosion_px", 1),
    ).clip(aoi)
    logger.info("Label image built (%d -> %d), erosion=%d px",
                bt["before_year"], bt["after_year"], lab.get("erosion_px", 1))

    # bitemporal predictor stack (built once, sampled per batch)
    stack = build_bitemporal_stack(aoi, cfg)
    logger.info("Bitemporal stack: %d feature bands",
                stack.bandNames().size().getInfo())

    # export one batch at a time: a different seed per batch -> disjoint samples
    for b in range(n_batches):
        samples = sample_training_table(
            stack=stack, label_image=label_img, aoi=aoi,
            n_per_class=per_batch, scale=smp["feature_scale"],
            seed=smp["seed"] + b,
        )
        task = ee.batch.Export.table.toDrive(
            collection=samples,
            description=f"features_{bt['before_year']}_{bt['after_year']}_b{b}",
            folder=cfg["gee"]["export_folder"],
            fileFormat="CSV",
        )
        task.start()
        logger.info("Batch %d/%d export started (%d pts/class) -> %s_b%d.csv",
                    b + 1, n_batches, per_batch,
                    f"features_{bt['before_year']}_{bt['after_year']}", b)

    logger.info("All %d batch exports started. Monitor: https://code.earthengine.google.com/tasks", n_batches)
    logger.info("When ALL complete, download the CSVs into data/raw/ and run 02_merge_tables.py")


if __name__ == "__main__":
    main()