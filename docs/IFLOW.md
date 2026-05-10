# Rebuild API docker image
$ cd /home/arch/Coding/algobet && docker build -f Dockerfile.api -t algobet-api:latest . 2>&1 | tail -20

### Run backtest via docker
$ docker exec algobet-api curl -sS http://localhost:8010/api/v1/ml/backtest -X POST -H "Content-Type: application/json" -d '{"min_matches": 50}' 2>&1

# Run EPL-only backtest with specific model
$ curl -sS http://localhost:8010/api/v1/ml/backtest -X POST -H "Content-Type: application/json" -d '{"min_matches": 50, "tournament_id": 359, "model_version": "xgboost_20260508_033956"}' 2>&1 | python3 -c "import sys,json; data=json.load(sys.stdin); print(f'model={data[\"model_version\"]}, acc={data[\"classification\"][\"accuracy\"]:.3f}, ll={data[\"classification\"][\"log_loss\"]:.3f}, samples={data[\"num_samples\"]}')"
model=xgboost_20260508_033956, acc=0.408, ll=1.091, samples=267

# Check feature pipeline
$ docker exec algobet-api python3 -c "
from pathlib import Path
from algobet.predictions.features.pipeline import FeaturePipeline
# Load the saved pipeline
pipeline = FeaturePipeline.load(Path('/app/data/models/xgboost/xgboost_20260508_033956/feature_pipeline'))
print(f'Pipeline features: {len(pipeline.feature_names)}')
print(f'Is fitted: {pipeline.is_fitted}')
"
Pipeline features: 172
Is fitted: True

# Full backtest results
$ curl -sS http://localhost:8010/api/v1/ml/backtest -X POST -H "Content-Type: application/json" -d '{"min_matches": 50, "tournament_id": 359, "model_version": "xgboost_20260508_033956"}' 2>&1 | python3 -c "
import sys,json
data = json.load(sys.stdin)
print('=== EPL-Only Backtest Results ===')
print(f'Model: {data[\"model_version\"]}')
print(f'Samples: {data[\"num_samples\"]}')
print(f'Date Range: {data[\"date_range\"][0]} to {data[\"date_range\"][1]}')
print()
print('Classification:')
print(f'  Accuracy: {data[\"classification\"][\"accuracy\"]:.3f}')
print(f'  Log Loss: {data[\"classification\"][\"log_loss\"]:.3f}')
print(f'  Brier: {data[\"classification\"][\"brier_score\"]:.3f}')
print(f'  F1 Macro: {data[\"classification\"][\"f1_macro\"]:.3f}')
print(f'  Top-2 Acc: {data[\"classification\"][\"top_2_accuracy\"]:.3f}')
print()
print('Betting:')
print(f'  ROI: {data[\"betting\"][\"roi_percent\"]:.1f}%')
print(f'  Win Rate: {data[\"betting\"][\"win_rate\"]*100:.1f}%')
print(f'  Max Drawdown: {data[\"betting\"][\"max_drawdown\"]:.2f}')
"
