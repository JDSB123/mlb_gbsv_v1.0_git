"""Quick check of synthetic data statistics after fixes."""
from mlbv1.data.synthetic_history import SyntheticHistoryGenerator
from mlbv1.data.loader import SyntheticDataLoader

gen = SyntheticHistoryGenerator()
df = gen.generate_seasons([2025])
totals = df["home_score"] + df["away_score"]
print("SyntheticHistoryGenerator:")
print(f"  Games: {len(df)}")
print(f"  Avg home: {df['home_score'].mean():.2f}")
print(f"  Avg away: {df['away_score'].mean():.2f}")
print(f"  Avg total: {totals.mean():.2f} (target ~8.6)")
print(f"  Home wins: {(df['home_score'] > df['away_score']).mean():.1%}")

loader = SyntheticDataLoader(num_games=1000)
df2 = loader.load()
totals2 = df2["home_score"] + df2["away_score"]
print("\nSyntheticDataLoader:")
print(f"  Games: {len(df2)}")
print(f"  Avg home: {df2['home_score'].mean():.2f}")
print(f"  Avg away: {df2['away_score'].mean():.2f}")
print(f"  Avg total: {totals2.mean():.2f} (target ~8.6)")
