CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    profile_key VARCHAR(80) NOT NULL UNIQUE,
    display_name VARCHAR(120) NOT NULL DEFAULT 'Local User',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profile_preferences (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER NOT NULL UNIQUE REFERENCES user_profiles(id) ON DELETE CASCADE,
    default_days_ahead INTEGER NOT NULL DEFAULT 1,
    min_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.55,
    min_ev DOUBLE PRECISION NOT NULL DEFAULT 0.05,
    favorite_bookie VARCHAR(120),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist_entries (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
    entry_type VARCHAR(20) NOT NULL,
    entry_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_watchlist_profile_entry UNIQUE (profile_id, entry_type, entry_id)
);

CREATE TABLE IF NOT EXISTS user_predictions (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    pick_1x2 VARCHAR(1),
    home_score INTEGER,
    away_score INTEGER,
    total_goals_line DOUBLE PRECISION,
    total_goals_pick VARCHAR(10),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_user_prediction_profile_match UNIQUE (profile_id, match_id)
);
