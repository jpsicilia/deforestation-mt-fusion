"""Tests that run WITHOUT Earth Engine (config + geometry logic).

    pytest tests/ -v

Keeping GEE out means these run in CI and on any machine.
"""
from __future__ import annotations

from shapely.geometry import box

from deforest_mt.config import load_config


def test_config_loads_and_has_bounds():
    cfg = load_config()
    b = cfg["study_area"]["bounds"]
    assert len(b) == 4, "bounds must be [W, S, E, N]"
    w, s, e, n = b
    assert w < e and s < n, "West<East and South<North required"


def test_utm_zone_is_22s_not_23s():
    # NW Mato Grosso (~59 W) -> UTM 22S (EPSG:31978). Guards the classic mix-up.
    cfg = load_config()
    assert cfg["study_area"]["utm_crs"] == "EPSG:31978"


def test_label_classes_are_disjoint():
    cfg = load_config()
    forest = set(cfg["labels"]["forest_classes"])
    anth = set(cfg["labels"]["anthropic_classes"])
    assert forest.isdisjoint(anth), "forest and anthropic class codes overlap"


def test_bitemporal_years_ordered():
    # before must precede after, or the change direction is meaningless.
    cfg = load_config()
    assert cfg["bitemporal"]["before_year"] < cfg["bitemporal"]["after_year"]


def test_deter_window_within_reason():
    # DETER comparison window should bracket the bitemporal years.
    cfg = load_config()
    assert cfg["deter"]["window_start"] < cfg["deter"]["window_end"]


def test_spatial_norm_percentile_valid():
    cfg = load_config()
    p = cfg["spatial_norm"]["percentile"]
    assert 50 <= p <= 99, "percentile for spatial normalisation should be high (e.g. 90)"


def test_aoi_area_is_plausible():
    cfg = load_config()
    w, s, e, n = cfg["study_area"]["bounds"]
    approx_km2 = box(w, s, e, n).area * 111 * 111
    assert 5_000 < approx_km2 < 100_000, f"AOI area looks off: {approx_km2:.0f} km2"
