#!/usr/bin/env python
"""Test Lahman data enrichment."""
import logging

from src.mlbv1.data.historical_enrichment import LahmanDataEnricher

logging.basicConfig(level=logging.INFO)

print("Fetching Lahman pitcher stats (2024)...")
pitcher_stats = LahmanDataEnricher.get_pitcher_stats(2024)
print(f"\nLoaded {len(pitcher_stats)} pitcher records")
print("\nSample pitcher stats (2024):")
print(pitcher_stats.head(10))
if not pitcher_stats.empty:
    print(
        f"\nERA range: {pitcher_stats['era'].min():.2f} - {pitcher_stats['era'].max():.2f}"
    )
    print(f"Avg ERA: {pitcher_stats['era'].mean():.2f}")
