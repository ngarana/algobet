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
