"""Training-data loading, splitting, feature generation, and caching."""

import numpy as np
from numpy.typing import NDArray

from algobet.predictions.training.split import (
    ExpandingWindowSplitter,
    SeasonAwareSplitter,
    TemporalSplitter,
    encode_targets,
)


class DataPreparationMixin:
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
        from algobet.predictions.features.pipeline import prepare_match_dataframe

        # Get historical matches with optional date and filter constraints
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

        # Check minimum matches requirement if specified
        min_matches = getattr(self.config, "min_matches", None)
        if min_matches and len(matches) < min_matches:
            raise ValueError(
                f"Insufficient matches: {len(matches)} < {min_matches}. "
                "Adjust date range or reduce minimum matches requirement."
            )

        # Convert to DataFrame
        matches_df = prepare_match_dataframe(matches)

        # Preload all team match history and H2H data into memory to avoid
        # per-match DB queries during feature generation (N+1 → 2 bulk queries)
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

        # Preload standings for tournament-season pairs
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

        # Add result column
        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )

        # Split data using configured strategy
        if self.config.split_strategy == "expanding_window":
            splitter = ExpandingWindowSplitter(
                min_train_size=self.config.min_train_size,
                val_size=self.config.ew_val_size,
                test_size=self.config.ew_test_size,
                step_size=self.config.step_size,
            )
        elif self.config.split_strategy == "season_aware":
            splitter = SeasonAwareSplitter(
                train_seasons=self.config.train_seasons,
                val_seasons=self.config.val_seasons,
                test_seasons=self.config.test_seasons,
            )
        else:
            splitter = TemporalSplitter(
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                gap_days=self.config.gap_days,
            )

        splits = list(splitter.split(matches_df))
        split = splits[0]  # Use the first (or only) split

        # Encode targets
        y = encode_targets(matches_df["result"].values)
        y_train = y[split.train_indices]
        y_val = y[split.val_indices]
        y_test = y[split.test_indices]

        # Fit pipeline on training data only, then transform all subsets
        train_df = matches_df.iloc[split.train_indices]
        val_df = matches_df.iloc[split.val_indices]
        test_df = matches_df.iloc[split.test_indices]
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df

        X_train = self.feature_pipeline.fit_transform(train_df, self.repo)
        self._train_raw_features = self.feature_pipeline.last_raw_features
        X_val = self.feature_pipeline.transform(val_df, self.repo)
        self._val_raw_features = self.feature_pipeline.last_raw_features
        X_test = self.feature_pipeline.transform(test_df, self.repo)
        self._test_raw_features = self.feature_pipeline.last_raw_features

        # Cache raw features if enabled, reusing frames already produced while
        # fitting and transforming the split data.
        if self.config.use_feature_cache:
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
