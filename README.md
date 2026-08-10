# Deforestation detection in NW Mato Grosso — Sentinel-1 SAR + Sentinel-2 optical fusion

A **bitemporal Random Forest** that fuses **Sentinel-1 (SAR)** and **Sentinel-2
(optical)**, with **Hamunyela spatial normalisation**, to detect deforestation on
the active clearing front of northwestern Mato Grosso (Amazon biome —
Colniza/Aripuanã, the region that leads Brazil's Amazon deforestation ranking),
and compares the result against Brazil's operational **DETER** alerts.

**Central question.** Under clouds — when the optical DETER system is blind — does
learned SAR+optical fusion recover clearing events that optical monitoring misses,
with fewer false positives than either sensor alone, under spatially correct
validation? This is a **proof-of-concept at limited scale**, not an operational
alert system.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full rationale behind every decision.

## Approach in one picture

```
MapBiomas transitions ──► labels (stable forest / deforestation)   [DETER held out]
Sentinel-2 (clouds masked) ─┐
Sentinel-1 (SAR)  ──────────┤─► features ─► spatial normalisation ─► bitemporal
                            │                (Hamunyela 2016)         (2022 vs 2023)
                            └───────────────────────────────────────────┐
                                                                         ▼
                        Random Forest (S2 / S1 / fusion)  ─►  deforestation map
                                                                         │
        spatial block CV + cloud-split eval + DETER overlay ◄────────────┘
```

## Reproduce in 5 steps

```bash
# 1. Environment
conda env create -f environment.yml && conda activate deforest-mt
pip install -e .                      # makes `deforest_mt` importable

# 2. Authenticate Earth Engine once
earthengine authenticate

# 3. Set your GEE project id (and optionally refine the AOI) in config/config.yaml

# 4. Verify everything works
python scripts/00_check_setup.py

# 5. (Phase 1 onward) build composites, features, and the training table
python scripts/01_build_features.py
```

Run tests with `pytest tests/`.

## Repo layout

```
config/          study-area bounds, years, class codes, normalisation & DETER params
src/deforest_mt/ importable package
  gee/           auth, aoi, sentinel2, sentinel1, labels, spatial_norm, features
  modeling/      Random Forest, spatial block CV, DETER comparison, evaluation
scripts/         runnable entry points (00_check, 01_build_features, ...)
notebooks/       exploration only (logic lives in src/)
tests/           GEE-free unit tests
docs/            DESIGN.md — decisions and rationale
data/ results/   git-ignored
```

## Key references (verified from primary source)

Doblas 2022 (DETER-R) · Ferrari 2023 (fusion, cloud-split eval) · Heckel 2020
(S1/S2/fusion + spatial CV) · Souza 2020 (MapBiomas) · Roberts 2017 (spatial CV) ·
Breiman 2001 (RF) · Haralick 1973 (GLCM) · Hamunyela 2016 (spatial normalisation) ·
Hamunyela 2017 (space-time features) · DeVries 2015 (regrowth) · Shimizu 2019
(RF fusion, 3 scenarios).

Author: José Pablo Pérez Cicilia · ILATIT/UNILA, Foz do Iguaçu–PR
