"""Phase 2 — train Random Forest and compare scenarios (local, no GEE).

    python scripts/03_train_rf.py

1. load the merged training table
2. evaluate 6 scenarios: {S2, S1, fusion} x {with, without spatial norm}
3. learning curve on the best scenario (how many points are enough?)
4. permutation feature importance (which features actually matter?)

Everything is saved to results/tables and results/figures. This is the first
real answer to "does the fusion + spatial normalisation help?".
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from deforest_mt.config import DATA_RAW, RESULTS_DIR, load_config
from deforest_mt.modeling.random_forest import (
    load_table, run_all_scenarios, select_scenario,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("phase2")


def learning_curve(df, sensor, with_snorm, seed=42):
    """F1 vs training size — tells us if more points would still help."""
    cols = select_scenario(df, sensor, with_snorm)
    X, y = df[cols].to_numpy(), df["label"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.7)
    tr, te = idx[:cut], idx[cut:]

    sizes, f1s = [], []
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        n = max(50, int(len(tr) * frac))
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
        clf.fit(X[tr[:n]], y[tr[:n]])
        f1s.append(f1_score(y[te], clf.predict(X[te]), pos_label=1, zero_division=0))
        sizes.append(n)
    return sizes, f1s


def feature_importance(df, sensor, with_snorm, seed=42, top=20):
    """Permutation importance (not Gini) — which features the model really uses."""
    cols = select_scenario(df, sensor, with_snorm)
    X, y = df[cols].to_numpy(), df["label"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.7)
    tr, te = idx[:cut], idx[cut:]

    clf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    imp = permutation_importance(
        clf, X[te], y[te], scoring="f1", n_repeats=10, random_state=seed, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1][:top]
    return [(cols[i], imp.importances_mean[i]) for i in order]


def main() -> None:
    cfg = load_config()
    bt = cfg["bitemporal"]
    table = DATA_RAW / f"training_table_{bt['before_year']}_{bt['after_year']}.csv"

    df = load_table(str(table))
    logger.info("Class balance:\n%s", df["label"].value_counts().to_string())

    # 1. six scenarios
    logger.info("=== Scenario comparison (random split, sanity pass) ===")
    results = run_all_scenarios(df, n_estimators=500, seed=cfg["sampling"]["seed"])
    out_tbl = RESULTS_DIR / "tables" / "scenario_comparison.csv"
    results.to_csv(out_tbl, index=False)
    logger.info("Saved -> %s\n%s", out_tbl, results.to_string(index=False))

    # 2. learning curve on fusion + snorm
    sizes, f1s = learning_curve(df, "fusion", True, seed=cfg["sampling"]["seed"])
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, f1s, "o-")
    plt.xlabel("training points"); plt.ylabel("F1 (deforestation)")
    plt.title("Learning curve — fusion + spatial norm"); plt.grid(True, alpha=0.3)
    fig1 = RESULTS_DIR / "figures" / "learning_curve.png"
    plt.savefig(fig1, dpi=120, bbox_inches="tight"); plt.close()
    logger.info("Learning curve F1: %s -> %s", [f"{f:.3f}" for f in f1s], fig1)

    # 3. feature importance on fusion + snorm
    fi = feature_importance(df, "fusion", True, seed=cfg["sampling"]["seed"])
    logger.info("=== Top features (permutation importance) ===")
    for name, val in fi:
        logger.info("  %-30s %.4f", name, val)
    pd.DataFrame(fi, columns=["feature", "importance"]).to_csv(
        RESULTS_DIR / "tables" / "feature_importance.csv", index=False)

    logger.info("Phase 2 done. Check results/tables and results/figures.")


if __name__ == "__main__":
    main()