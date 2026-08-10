"""Earth Engine initialisation — one entry point for every script.

Run `earthengine authenticate` once in your terminal before first use.
"""
from __future__ import annotations

import logging

import ee

logger = logging.getLogger(__name__)


def init_ee(project: str) -> None:
    """Initialise Earth Engine for a Cloud project.

    Args:
        project: Your GEE / Google Cloud project id (from config.yaml).

    Raises:
        ee.EEException: if authentication has never been run on this machine.
    """
    try:
        ee.Initialize(project=project)
        logger.info("Earth Engine initialised (project=%s)", project)
    except ee.EEException:
        logger.warning("EE not initialised; run `earthengine authenticate` first.")
        raise
