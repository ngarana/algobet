docker exec -i algobet-db psql -U algobet -d football -c "
-- Delete duplicate matches, keeping only the most recent one for each team pair on each date
DELETE FROM matches
WHERE id IN (
    SELECT id
    FROM (
        SELECT
            id,
            ROW_NUMBER() OVER (
                PARTITION BY home_team_id, away_team_id, DATE(match_date)
                ORDER BY created_at DESC
            ) as rn
        FROM matches
        WHERE DATE(match_date) = '2026-04-19'
    ) ranked
    WHERE rn > 1
);
"
