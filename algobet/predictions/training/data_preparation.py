"""Training-data loading, splitting, feature generation, and caching."""

import numpy as np
from numpy.typing import NDArray

from algobet.predictions.training.split import (
    ExpandingWindowSplitter,
    SeasonAwareSplitter,
    TemporalSplitter,
    WalkForwardSplitter,
    encode_targets,
)


class DataPreparationMixin:
    def _load_training_matches_dataframe(self):
        """Load, preload, and normalize historical matches for training."""
        from algobet.predictions.features.pipeline import prepare_match_dataframe

        matches = self.repo.get_historical_matches(
            min_date=self.config.start_date,
            max_date=self.config.end_date,
            tournament_ids=self.config.tournament_ids,
            team_ids=self.config.team_ids,
            require_results=True,
            min_total_goals=self.config.min_total_goals,
            max_total_goals=self.config.max_total_goals,
            venue_filter=self.config.venue_filter,
        )

        if not matches:
            raise ValueError("No historical matches found for training")

        min_matches = getattr(self.config, "min_matches", None)
        if min_matches and len(matches) < min_matches:
            raise ValueError(
                f"Insufficient matches: {len(matches)} < {min_matches}. "
                "Adjust date range or reduce minimum matches requirement."
            )

        matches_df = prepare_match_dataframe(matches)

        all_team_ids = list(
            set(
                matches_df["home_team_id"].tolist()
                + matches_df["away_team_id"].tolist()
            )
        )
        max_match_date = matches_df["match_date"].max()
        self.repo.preload_team_matches(all_team_ids, before_date=max_match_date)
        team_pairs = list(
            zip(
                matches_df["home_team_id"].tolist(),
                matches_df["away_team_id"].tolist(),
                strict=False,
            )
        )
        self.repo.preload_h2h_matches(team_pairs, before_date=max_match_date)

        tournament_season_pairs = list(
            set(
                zip(
                    matches_df["tournament_id"].tolist(),
                    matches_df["season_id"].tolist(),
                    strict=False,
                )
            )
        )
        self.repo.preload_season_standings(
            tournament_season_pairs, before_date=max_match_date
        )

        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )
        return matches_df

    def _build_splitter(self):
        """Create the configured temporal splitter."""
        if self.config.split_strategy == "expanding_window":
            return ExpandingWindowSplitter(
                min_train_size=self.config.min_train_size,
                val_size=self.config.ew_val_size,
                test_size=self.config.ew_test_size,
                step_size=self.config.step_size,
            )
        if self.config.split_strategy == "season_aware":
            return SeasonAwareSplitter(
                train_seasons=self.config.train_seasons,
                val_seasons=self.config.val_seasons,
                test_seasons=self.config.test_seasons,
            )
        if self.config.split_strategy == "walk_forward":
            return WalkForwardSplitter(
                train_seasons=self.config.train_seasons,
                val_seasons=self.config.val_seasons,
                test_seasons=self.config.test_seasons,
            )
        return TemporalSplitter(
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            gap_days=self.config.gap_days,
        )

    def _prepare_data_for_split(
        self,
        matches_df,
        split,
        *,
        cache_features: bool,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
        """Prepare matrices for one split and store split state on the pipeline."""
        y = encode_targets(matches_df["result"].values)
        y_train = y[split.train_indices]
        y_val = y[split.val_indices]
        y_test = y[split.test_indices]

        train_df = matches_df.iloc[split.train_indices]
        val_df = matches_df.iloc[split.val_indices]
        test_df = matches_df.iloc[split.test_indices]
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df

        raw_cache = getattr(self, "_prepared_raw_features", None)

        if raw_cache is not None:

            def raw_for_split(frame):
                match_ids = frame["id"].tolist()
                missing = [
                    match_id
                    for match_id in match_ids
                    if match_id not in raw_cache.index
                ]
                if missing:
                    raise ValueError(
                        "Prepared raw feature cache is missing match ids: "
                        f"{missing[:5]}"
                    )
                return raw_cache.loc[match_ids]

            self._train_raw_features = raw_for_split(train_df)
            self._val_raw_features = raw_for_split(val_df)
            self._test_raw_features = raw_for_split(test_df)

            X_train = self.feature_pipeline.fit_transform_raw_features(
                self._train_raw_features,
                y_train,
            )
            X_val = self.feature_pipeline.transform_raw_features(self._val_raw_features)
            X_test = self.feature_pipeline.transform_raw_features(
                self._test_raw_features
            )
        else:
            X_train = self.feature_pipeline.fit_transform(train_df, self.repo)
            self._train_raw_features = self.feature_pipeline.last_raw_features
            X_val = self.feature_pipeline.transform(val_df, self.repo)
            self._val_raw_features = self.feature_pipeline.last_raw_features
            X_test = self.feature_pipeline.transform(test_df, self.repo)
            self._test_raw_features = self.feature_pipeline.last_raw_features

        if cache_features and self.config.use_feature_cache:
            try:
                import pandas as pd

                raw_frames = [
                    frame
                    for frame in (
                        self._train_raw_features,
                        self._val_raw_features,
                        self._test_raw_features,
                    )
                    if frame is not None
                ]
                raw_features = pd.concat(raw_frames) if raw_frames else pd.DataFrame()
                from algobet.predictions.features.store import features_to_store_format

                features_list = features_to_store_format(
                    raw_features,
                    schema_version=self.config.feature_schema_version,
                )
                savepoint = self.session.begin_nested()
                self.feature_store.store_bulk(features_list)
                savepoint.commit()
            except Exception:
                try:
                    savepoint.rollback()
                except Exception:
                    self.session.rollback()

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _prepare_data(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
        """Prepare training data from database.

        Steps:
        1. Load historical matches
        2. Generate raw features once for all matches
        3. Split by temporal indices
        4. Fit transformers on training subset only
        5. Transform all three subsets
        6. Cache raw features for reproducibility
        """
        matches_df = self._load_training_matches_dataframe()
        self._prepared_raw_features = self.feature_pipeline.generate_raw(
            matches_df,
            self.repo,
        )
        splitter = self._build_splitter()
        splits = list(splitter.split(matches_df))
        if not splits:
            raise ValueError("No valid temporal splits could be generated")

        self._prepared_matches_df = matches_df
        self._prepared_splits = splits

        # For walk-forward, train the saved artifact on the most recent fold.
        # Cross-fold averages are computed separately in the runner.
        split = (
            splits[-1] if self.config.split_strategy == "walk_forward" else splits[0]
        )

        return self._prepare_data_for_split(matches_df, split, cache_features=True)
