"""Live feature preparation with historical context (cold-start prevention)."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from mlbv1.data.loader import MLBStatsAPILoader
from mlbv1.data.preprocessor import ProcessedData, preprocess
from mlbv1.features.engineer import FeatureSet, engineer_features

logger = logging.getLogger(__name__)


def prepare_live_features_with_history(
    live_df: pd.DataFrame,
    config: Any,
) -> tuple[ProcessedData, FeatureSet, int]:
    """Build live prediction features using recent historical context.

    This prevents early-season / slate-only cold starts where rolling features
    collapse to zero when only today's games are present.

    Returns (processed_live, feature_live, hist_rows_count).
    """
    history_days = int(os.getenv("LIVE_CONTEXT_DAYS", "120"))
    live = live_df.copy().reset_index(drop=True)
    live["_is_live"] = 1

    hist = pd.DataFrame()
    end = datetime.now(tz=UTC)
    start = end - pd.Timedelta(days=history_days)
    try:
        hist_loader = MLBStatsAPILoader(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        hist = hist_loader.load()
        logger.info("Loaded %d historical context games for live features", len(hist))
    except Exception as exc:
        logger.warning("Historical context load failed; using live-only features: %s", exc)

    hist_rows = 0
    if not hist.empty:
        hist = hist.copy()
        hist_rows = len(hist)
        hist["_is_live"] = 0
        for col in live.columns:
            if col not in hist.columns:
                hist[col] = pd.NA
        for col in hist.columns:
            if col not in live.columns:
                live[col] = pd.NA
        combined = pd.concat([hist, live], ignore_index=True, sort=False)
    else:
        combined = live

    processed_all = preprocess(combined)
    features_all = engineer_features(
        processed_all.features,
        short_window=config.features.rolling_window_short,
        long_window=config.features.rolling_window_long,
    )

    live_mask = (
        processed_all.features["_is_live"].fillna(0).astype(int) == 1
        if "_is_live" in processed_all.features.columns
        else pd.Series(
            [True] * len(processed_all.features),
            index=processed_all.features.index,
        )
    )

    live_features_df = (
        processed_all.features.loc[live_mask]
        .drop(columns=["_is_live"], errors="ignore")
        .reset_index(drop=True)
    )
    live_target_df = processed_all.target.loc[live_mask].reset_index(drop=True)
    live_metadata_df = processed_all.metadata.loc[live_mask].reset_index(drop=True)
    live_targets_df = (
        processed_all.targets.loc[live_mask].reset_index(drop=True)
        if processed_all.targets is not None
        else None
    )
    live_X = features_all.X.loc[live_mask].reset_index(drop=True)

    processed_live = ProcessedData(
        features=live_features_df,
        target=live_target_df,
        metadata=live_metadata_df,
        targets=live_targets_df,
    )
    feature_live = FeatureSet(X=live_X, feature_names=list(live_X.columns))
    return processed_live, feature_live, hist_rows
