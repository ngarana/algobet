docker exec algobet-db psql -U algobet -d football -c "
SELECT
  COUNT(*) as total_finished,
  SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as home_wins,
  SUM(CASE WHEN home_score = away_score THEN 1 ELSE 0 END) as draws,
  SUM(CASE WHEN home_score < away_score THEN 1 ELSE 0 END) as away_wins,
  ROUND(100.0*SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END)/COUNT(*),1) as h_pct,
  ROUND(100.0*SUM(CASE WHEN home_score = away_score THEN 1 ELSE 0 END)/COUNT(*),1) as d_pct,
  ROUND(100.0*SUM(CASE WHEN home_score < away_score THEN 1 ELSE 0 END)/COUNT(*),1) as a_pct
FROM matches
WHERE status='FINISHED' AND home_score IS NOT NULL;
"
-- Tournament breakdown
docker exec algobet-db psql -U algobet -d football -c "
-- Tournament breakdown
SELECT t.name, m.tournament_id, COUNT(*) as cnt, MIN(m.match_date)::date as earliest, MAX(m.match_date)::date as latest
FROM matches m JOIN tournaments t ON m.tournament_id = t.id
WHERE m.status='FINISHED'
GROUP BY t.name, m.tournament_id ORDER BY cnt DESC;
"

-- Season breakdown for EPL (tournament_id=359)
docker exec algobet-db psql -U algobet -d football -c "
-- Season breakdown for EPL (tournament_id=359)
SELECT s.name as season, COUNT(*) as matches,
  MIN(m.match_date)::date as start_dt, MAX(m.match_date)::date as end_dt,
  ROUND(100.0*SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END)/COUNT(*),1) as h_pct,
  ROUND(100.0*SUM(CASE WHEN home_score = away_score THEN 1 ELSE 0 END)/COUNT(*),1) as d_pct,
  ROUND(100.0*SUM(CASE WHEN home_score < away_score THEN 1 ELSE 0 END)/COUNT(*),1) as a_pct
FROM matches m JOIN seasons s ON m.season_id = s.id
WHERE m.status='FINISHED' AND m.tournament_id=359
GROUP BY s.name, m.season_id ORDER BY start_dt;
"


-- Check how many matches per team in training window (for form feature quality)
docker exec algobet-db psql -U algobet -d football -c "
-- Check how many matches per team in training window (for form feature quality)
SELECT t.name, COUNT(*) as matches
FROM matches m JOIN teams t ON m.home_team_id = t.id OR m.away_team_id = t.id
WHERE m.status='FINISHED' AND m.tournament_id=359 AND m.match_date <= '2025-05-31'
GROUP BY t.name ORDER BY matches DESC LIMIT 25;

-- Check xG coverage by season for EPL
SELECT s.name as season,
  COUNT(*) as total_matches,
  SUM(CASE WHEN ms.home_xg IS NOT NULL THEN 1 ELSE 0 END) as with_xg,
  ROUND(100.0 * SUM(CASE WHEN ms.home_xg IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 0) as xg_pct
FROM matches m
JOIN seasons s ON m.season_id = s.id
LEFT JOIN match_statistics ms ON m.id = ms.match_id
WHERE m.status='FINISHED' AND m.tournament_id=359
GROUP BY s.name, m.season_id ORDER BY MIN(m.match_date);
"

-- Training data: what training uses (tournament_id=359, end_date 2025-05-31)
docker exec algobet-db psql -U algobet -d football -c "
-- Training data: what training uses (tournament_id=359, end_date 2025-05-31)
SELECT
  COUNT(*) as training_matches,
  MIN(match_date)::date as earliest,
  MAX(match_date)::date as latest
FROM matches
WHERE status='FINISHED' AND tournament_id=359 AND match_date <= '2025-05-31' AND home_score IS NOT NULL;

-- Backtest data: what backtest uses (2025-11-01 to 2026-05-10)
SELECT
  COUNT(*) as backtest_matches,
  ROUND(100.0*SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END)/COUNT(*),1) as h_pct,
  ROUND(100.0*SUM(CASE WHEN home_score = away_score THEN 1 ELSE 0 END)/COUNT(*),1) as d_pct,
  ROUND(100.0*SUM(CASE WHEN home_score < away_score THEN 1 ELSE 0 END)/COUNT(*),1) as a_pct
FROM matches
WHERE status='FINISHED' AND tournament_id=359
  AND match_date >= '2025-11-01' AND match_date <= '2026-05-10'
  AND home_score IS NOT NULL AND odds_home IS NOT NULL;



-- Enriched stats coverage for training vs backtest periods
SELECT
  CASE WHEN m.match_date < '2025-06-01' THEN 'training' ELSE 'backtest' END as period,
  COUNT(*) as total_matches,
  SUM(CASE WHEN ms.home_xg IS NOT NULL THEN 1 ELSE 0 END) as has_xg,
  SUM(CASE WHEN pms.match_id IS NOT NULL THEN 1 ELSE 0 END) as has_player_stats
FROM matches m
LEFT JOIN match_statistics ms ON m.id = ms.match_id
LEFT JOIN (SELECT DISTINCT match_id FROM player_match_stats) pms ON m.id = pms.match_id
WHERE m.status='FINISHED' AND m.tournament_id=359 AND m.home_score IS NOT NULL
GROUP BY period ORDER BY period;


docker exec algobet-db psql -U algobet -d football -c "WITH scoped AS (SELECT CASE WHEN EXTRACT(MONTH FROM match_date) < 7 THEN EXTRACT(YEAR FROM match_date)::int - 1 ELSE EXTRACT(YEAR FROM match_date)::int END AS football_season, * FROM matches WHERE tournament_id IN (359,545,98,28,123) AND match_date >= DATE '2012-07-01' AND match_date <= TIMESTAMP '2025-06-30 23:59:59' AND status='FINISHED' AND home_score IS NOT NULL AND away_score IS NOT NULL) SELECT football_season, COUNT(*) AS matches, COUNT(*) FILTER (WHERE opening_odds_home IS NOT NULL AND opening_odds_draw IS NOT NULL AND opening_odds_away IS NOT NULL) AS opening_1x2, COUNT(*) FILTER (WHERE closing_odds_home IS NOT NULL AND closing_odds_draw IS NOT NULL AND closing_odds_away IS NOT NULL) AS closing_1x2, ROUND(100.0 * COUNT(*) FILTER (WHERE closing_odds_home IS NOT NULL AND closing_odds_draw IS NOT NULL AND closing_odds_away IS NOT NULL) / NULLIF(COUNT(*), 0), 2) AS closing_pct FROM scoped GROUP BY football_season ORDER BY football_season;


curl -sS -X POST http://localhost:8010/api/v1/ml/train -H 'Content-Type: application/json' -d '{"model_type":"market_mediation","description":"Top5 selective market mediation with true closing-line CLV","activate":false,"tournament_ids":[359,545,98,28,123],"start_date":"2012-07-01","end_date":"2025-06-30T23:59:59","split_strategy":"walk_forward","train_seasons":8,"val_seasons":1,"test_seasons":1,"feature_groups":["team_form","head_to_head","temporal","standings","enriched_stats","draw_signals","matchup_interaction","player_quality","market_mediation"],"closing_odds_required":true,"production_lane":"dual","taken_odds_snapshot":"opening","min_expected_clv":0.005,"min_positive_clv_probability":0.55}



UV_CACHE_DIR=/tmp/uv-cache POSTGRES_HOST=localhost POSTGRES_PORT=5432 POSTGRES_USER=algobet POSTGRES_PASSWORD=password POSTGRES_DB=football uv run python - <<'PY'
from datetime import datetime
from pathlib import Path
import traceback
import algobet.teams.models  # noqa: F401
import algobet.predictions.models.base  # noqa: F401
from algobet.infrastructure.database import session_scope
from algobet.predictions.training.pipeline import TrainingConfig, TrainingPipeline

config = TrainingConfig(
    model_type='market_mediation',
    description='debug market mediation',
    tournament_ids=[359,545,98,28,123],
    start_date=datetime(2021,7,1),
    end_date=datetime(2025,6,30,23,59,59),
    split_strategy='walk_forward',
    train_seasons=1,
    val_seasons=1,
    test_seasons=1,
    feature_groups=['team_form','head_to_head','temporal','standings','enriched_stats','draw_signals','matchup_interaction','player_quality','market_mediation'],
    closing_odds_required=True,
    production_lane='dual',
    taken_odds_snapshot='opening',
    min_expected_clv=0.005,
    min_positive_clv_probability=0.55,
)
with session_scope() as session:
    try:
        result = TrainingPipeline(config=config, session=session, models_path=Path('/tmp/algobet-debug-models')).run()
        print('OK', result.model_version, result.test_metrics)
    except Exception:
        traceback.print_exc()
PY
