"""Quick model prediction range check."""
from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.models.predictor import load_model, predict

df = SyntheticDataLoader(num_games=50, seed=99).load()
proc = preprocess(df)
feats = engineer_features(proc.features)

models = ["random_forest", "ridge_regression", "xgboost", "lightgbm"]
for name in models:
    m = load_model(f"artifacts/models/{name}.pkl")
    result = predict(m, feats.X, df)
    er = result.expected_runs
    avg_h = er["home_score"].mean()
    avg_a = er["away_score"].mean()
    avg_t = (er["home_score"] + er["away_score"]).mean()
    max_t = (er["home_score"] + er["away_score"]).max()
    mp = result.market_probabilities
    avg_ml = mp["home_ml_prob"].mean()
    print(f"{name:20s}: avg_total={avg_t:.2f} max={max_t:.2f} home_ml={avg_ml:.3f}")

print(f"\nExpected: avg_total ~8.6, home_ml ~0.52")
