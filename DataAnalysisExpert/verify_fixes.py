"""Quick verification of synthetic data fixes."""
import sys
sys.path.insert(0, "src")
import numpy as np
from mlbv1.data.synthetic_history import SyntheticHistoryGenerator
from mlbv1.features.engineer import engineer_features

gen = SyntheticHistoryGenerator(seed=42)
synth = gen.generate_seasons([2023, 2024, 2025])

print(f"Total games: {len(synth)}")
print(f"Per year: {synth.groupby(synth['game_date'].dt.year).size().to_dict()}")
print(f"Home score mean: {synth['home_score'].mean():.2f}")
print(f"Away score mean: {synth['away_score'].mean():.2f}")
print(f"Total mean: {(synth['home_score'] + synth['away_score']).mean():.2f}")

# Check total_runs leakage
corr_home = synth["total_runs"].corr(synth["home_score"])
corr_away = synth["total_runs"].corr(synth["away_score"])
print(f"\ntotal_runs ↔ home_score correlation: {corr_home:.3f} (was 0.697)")
print(f"total_runs ↔ away_score correlation: {corr_away:.3f} (was 0.623)")

# Kelly check
from scripts.pick_sheet import _kelly_fraction
# Example: 70% model prob, +150 odds (decimal 2.5)
k = _kelly_fraction(0.70, 2.5)
print(f"\nKelly(70%, +150): {k*100:.1f}% (was {max(0, (0.70*1.5 - 0.30)/1.5)*100:.1f}%)")
# Full Kelly would be (0.70*1.5 - 0.30)/1.5 = 50%, quarter = 12.5%
