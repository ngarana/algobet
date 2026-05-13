-- Widen feature_schema_version columns from VARCHAR(20) to VARCHAR(50)
-- to accommodate longer schema version strings like 'v3.0_epl_feature_tuning'.

ALTER TABLE model_versions ALTER COLUMN feature_schema_version TYPE VARCHAR(50);
ALTER TABLE model_features ALTER COLUMN feature_schema_version TYPE VARCHAR(50);
