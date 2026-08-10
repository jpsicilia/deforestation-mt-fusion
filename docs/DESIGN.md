# Design decisions — deforestation-mt-fusion

This document records **why** the project is built the way it is. Every decision
here was settled deliberately; do not change one without re-reading its rationale.

## The question

> In the active clearing front of NW Mato Grosso (Colniza/Aripuanã, Amazon
> biome), does a **bitemporal Random Forest fusing Sentinel-1 SAR + Sentinel-2
> optical**, with **Hamunyela spatial normalisation**, detect deforestation that
> the optical DETER system misses under clouds — with fewer false positives than
> either sensor alone — under spatially correct validation?

It is a **proof-of-concept / feasibility study at limited scale**, not an
operational alert system.

## Scope: Camino A (bitemporal), NOT Camino B (time series)

- **Chosen (A):** two dry-season composites (before = 2022, after = 2023),
  compare the change. Simple, interpretable, executable within the deadline.
- **Rejected for now (B):** dense time-series changepoint / harmonic features /
  space-time features from local data cubes (Reiche, Shimizu, Hamunyela 2017).
  Superior but months of work + Dask/out-of-memory cubes → **future work**.

## Labels (anti-circularity)

- Training/testing labels from **MapBiomas** annual transitions:
  forest(t)→forest(t+1) = stable forest (0); forest(t)→anthropic(t+1) =
  deforestation (1). Everything else masked.
- **DETER is held out entirely** as an independent comparison reference — never
  used for training. Avoids the circularity trap.
- Labels inherit MapBiomas error (~95% accuracy in Amazon) — declared as a limitation.

## Predictors

- **Sentinel-2** (optical): B2 B3 B4 B8 B11 B12 + NDVI EVI NBR NDWI SAVI. Cloud
  masked (Cloud Score+). 13-band / 10 m — better than DETER's CBERS (4 band / 64 m).
- **Sentinel-1** (SAR): VV, VH, VV/VH ratio + GLCM texture. GEE GRD is already
  border-noise/thermal/calibration/terrain corrected (Ferrari 2023); temporal
  median + focal median for speckle.
- **Bitemporal:** features from the *before* and *after* composites plus their
  **differences** — detects the change, not a single-date state. Addresses the
  regrowth problem (a regrown pixel still shows a drop 2022→2023).

## Hamunyela spatial normalisation (the differentiator)

- Formula (Hamunyela et al. 2016, Eq. 1): `sNDVI = VI_pixel / VI_median`, where
  `VI_median` is the median of neighbours **above the 90th percentile** in a
  moving window (~25×25 px ≈ optimal at the Brazilian evergreen site).
- Rationale: drought/seasonality is **regional** (affects pixel and neighbours
  equally → cancels in the ratio ≈ 1); deforestation is **local** (amplified).
  Directly answers the "NDVI drops from drought" problem.
- Applied to **each composite separately**, to both optical (NDVI, NBR…) and SAR
  (VV, VH). Hamunyela's thesis Ch. 6 validates it on SAR too — but verify per
  data that it helps SAR rather than adding noise; if it hurts SAR, apply to optical only.

## What is compared with what (never mix states)

- The two composites (2022, 2023): **both normalised** → compared like-with-like.
- Final map vs DETER: **both binary** (0/1) → compared like-with-like.
- We never compare a normalised image against a raw one.

## Comparison against DETER

- We download DETER **polygons** (vectors) from TerraBrasilis — not images. No
  normalisation applies to DETER; it is the reference, used as-is.
- DETER Amazônia (Floresta) is **optical** (CBERS WFI/AWiFS, 56–64 m), **visual
  human interpretation** (~10 analysts, INPE Belém). Verified 2025. Not obsolete.
  SAR/fused branches (DETER-R/RT, Intenso) are partial or not publicly distributed.
- **Temporal alignment (critical):** compare only against DETER alerts **within
  the window** covered by the two composites (2022→2023). Alerts outside are
  excluded — they are not our error. This fixes the "deforestation happened after
  my image" invalid-comparison problem.
- **Map vs map, not image vs polygon:** rasterise all in-window DETER
  deforestation-class polygons into one binary reference map; overlay pixel-wise.
- Filters on DETER: temporal (in-window) + class (deforestation/clear-cut only).

## Validation

- **Spatial block cross-validation** (Roberts 2017), not random — avoids
  autocorrelation-inflated accuracy.
- Three scenarios: S2-only / S1-only / fusion, on the **same points**.
- Also: with vs without spatial normalisation → measures the differentiator.
- Sanity checks: learning curve, train/test gap, trivial baseline.
- **Money shot (Ferrari 2023):** metrics split by cloudy vs clear pixels.
- **Discrepancy verification:** a random sample of pixels where we and DETER
  disagree, checked by **visual interpretation** on high-resolution imagery
  (Planet NICFI ~4.7 m — free for the tropics; Google Earth; post-season clear
  Sentinel-2). This is the only way to say who was right. Resolution mismatch
  handled by deciding "corte/no corte at this location", a minimum mapping unit,
  and a position tolerance — declared as a limitation.

## Future work (cite, don't build now)

Time-series changepoint (Reiche/RADD/DETER-R), harmonic features + space-time
features from local data cubes (Shimizu 2019; Hamunyela 2017), InSAR coherence,
calibrated per-pixel uncertainty (conformal). These position the work at the
frontier without blowing the deadline.

## Verified key references (primary source)

Doblas 2022 (DETER-R) · Ferrari 2023 (transformer fusion, cloud-split eval) ·
Heckel 2020 (S1/S2/fusion + spatial CV) · Souza 2020 (MapBiomas) · Roberts 2017
(spatial CV) · Breiman 2001 (RF) · Haralick 1973 (GLCM) · Hamunyela 2016 (spatial
normalisation) · Hamunyela 2017 (space-time features) · DeVries 2015 (regrowth) ·
Shimizu 2019 (RF fusion, 3 scenarios).
