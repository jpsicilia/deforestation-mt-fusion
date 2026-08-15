"""Random Forest deforestation classifier (Phase 2) — runs locally, no GEE.

Loads the merged feature table and evaluates three sensor scenarios (S2-only,
S1-only, fusion), each with and without Hamunyela spatial normalisation, so the
differentiator's contribution is measured, not assumed.

Because the problem is imbalanced (~63:1 in reality) we train on a BALANCED
sample and report F1 / precision / recall for the deforestation class -- never
accuracy, which the majority class would inflate.

Column-subset scenarios (all from the same table):
    S2-only : optical temporal features (+ their diffs, + optical snorm)
    S1-only : SAR temporal features (+ their diffs, + SAR snorm)
    fusion  : everything
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)

# substrings that identify each sensor's columns
_S2_KEYS = ("NDVI", "EVI", "NBR", "NDWI", "SAVI")
_S1_KEYS = ("VV", "VH")
_LABEL = "label"
_DROP = ("longitude", "latitude", "lon", "lat", ".geo", "system:index")


def load_table(csv_path: str) -> pd.DataFrame:
    """Load the merged training table and drop non-feature columns."""
    df = pd.read_csv(csv_path)
    drop = [c for c in df.columns if c.lower() in _DROP]
    df = df.drop(columns=drop, errors="ignore")
    df = df.dropna().reset_index(drop=True)
    logger.info("Loaded %d rows, %d feature cols", len(df), df.shape[1] - 1)
    return df


def _is_snorm(col: str) -> bool:
    return col.endswith("_snorm") or "_snorm_" in col


def select_scenario(
    df: pd.DataFrame, sensor: str, with_snorm: bool
) -> list[str]:
    """Return the feature columns for a scenario.

    Args:
        df: full table.
        sensor: 's2', 's1' or 'fusion'.
        with_snorm: include spatially-normalised features or not.

    Returns:
        List of feature column names for this scenario.
    """
    feats = [c for c in df.columns if c != _LABEL]
    if sensor == "s2":
        cols = [c for c in feats if any(k in c for k in _S2_KEYS)]
    elif sensor == "s1":
        cols = [c for c in feats if any(k in c for k in _S1_KEYS)]
    else:  # fusion
        cols = list(feats)
    if not with_snorm:
        cols = [c for c in cols if not _is_snorm(c)]
    return cols


@dataclass
class ScenarioResult:
    sensor: str
    with_snorm: bool
    n_features: int
    precision: float
    recall: float
    f1: float


def evaluate_scenario(
    df: pd.DataFrame,
    sensor: str,
    with_snorm: bool,
    n_estimators: int = 500,
    seed: int = 42,
    test_frac: float = 0.3,
) -> ScenarioResult:
    """Train + evaluate one scenario with a random split (baseline check).

    NOTE: this random split is a first sanity pass. Spatially correct evaluation
    (block CV) comes in Phase 3 and is the number we report.
    """
    cols = select_scenario(df, sensor, with_snorm)
    X = df[cols].to_numpy()
    y = df[_LABEL].to_numpy()

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]

    clf = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])

    p, r, f1, _ = precision_recall_fscore_support(
        y[te], pred, labels=[1], average="binary", zero_division=0
    )
    logger.info("%-6s snorm=%-5s | feats=%3d | P=%.3f R=%.3f F1=%.3f",
                sensor, with_snorm, len(cols), p, r, f1)
    return ScenarioResult(sensor, with_snorm, len(cols), p, r, f1)


def run_all_scenarios(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Evaluate the 6 scenarios (3 sensors x with/without snorm)."""
    rows = []
    for sensor in ("s2", "s1", "fusion"):
        for with_snorm in (False, True):
            rows.append(evaluate_scenario(df, sensor, with_snorm, **kw))
    return pd.DataFrame([r.__dict__ for r in rows])