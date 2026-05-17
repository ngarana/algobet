"""Temporal data splitting for match prediction.

Provides time-aware data splitting strategies to prevent data leakage
in football match prediction models. Critical rule: never use future
data to predict past matches.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_SPLIT_SEASON_COLUMN = "_algobet_split_season"


def _season_sort_key(season: Any) -> int:
    season_str = str(season)
    try:
        return (
            int(season_str.split("/")[0])
            if "/" in season_str
            else int(float(season_str))
        )
    except (ValueError, TypeError):
        return 0


def _football_season_from_dates(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates)
    return parsed.dt.year - (parsed.dt.month < 7).astype(int)


def _should_use_calendar_season(df: pd.DataFrame, season_column: str) -> bool:
    if season_column not in df.columns:
        return True

    # In multi-league training, season_id is tournament-specific. Grouping by
    # it creates one pseudo-season per league-season instead of one football
    # season across the whole training frame.
    return "tournament_id" in df.columns and df["tournament_id"].dropna().nunique() > 1


def _with_split_season_column(
    df: pd.DataFrame,
    season_column: str,
    date_column: str,
    splitter_name: str,
) -> tuple[pd.DataFrame, str]:
    df = df.copy()

    if _should_use_calendar_season(df, season_column):
        df[_SPLIT_SEASON_COLUMN] = _football_season_from_dates(df[date_column])
        split_column = _SPLIT_SEASON_COLUMN
        if season_column in df.columns:
            logger.debug(
                "%s: using calendar football seasons from %s instead of %s",
                splitter_name,
                date_column,
                season_column,
            )
    else:
        split_column = season_column

    null_mask = df[split_column].isna()
    if null_mask.any():
        logger.warning(
            "%s: dropping %d row(s) with null %s before splitting",
            splitter_name,
            null_mask.sum(),
            split_column,
        )
        df = df[~null_mask].copy()

    return df, split_column


@dataclass
class TemporalSplit:
    """Represents a single temporal train/val/test split."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def val_size(self) -> int:
        return len(self.val_indices)

    @property
    def test_size(self) -> int:
        return len(self.test_indices)


class TemporalSplitter:
    """Time-aware data splitter for match prediction.

    Ensures that training data always precedes validation and test data
    to simulate real-world prediction scenarios where we predict future
    matches based on historical data.

    Example:
        >>> splitter = TemporalSplitter(
        ...     train_ratio=0.7,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15,
        ... )
        >>> splits = splitter.split(matches_df, date_column="match_date")
        >>> for split in splits:
        ...     X_train = X[split.train_indices]
        ...     X_val = X[split.val_indices]
        ...     X_test = X[split.test_indices]
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        date_column: str = "match_date",
        gap_days: int = 0,
    ) -> None:
        """Initialize temporal splitter.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            date_column: Name of the date column in DataFrame
            gap_days: Gap in days between train/val and val/test

        Raises:
            ValueError: If ratios don't sum to approximately 1.0
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.date_column = date_column
        self.gap_days = gap_days

    def split(
        self,
        df: pd.DataFrame,
        n_splits: int = 1,
    ) -> Iterator[TemporalSplit]:
        """Generate temporal train/val/test splits.

        Args:
            df: DataFrame with match data, sorted by date
            n_splits: Number of splits to generate (for rolling window CV)

        Yields:
            TemporalSplit objects with indices and date ranges
        """
        # Keep original row positions so callers can use the returned indices
        # against their own unsorted frame without drifting after sorting.
        df = df.sort_values(self.date_column)
        dates = pd.to_datetime(df[self.date_column])
        positions = df.index.to_numpy()

        n = len(df)

        if n_splits == 1:
            # Single split
            yield self._single_split(df, dates, positions, n)
        else:
            # Multiple rolling splits for time-series CV
            yield from self._rolling_splits(df, dates, positions, n, n_splits)

    def _single_split(
        self,
        df: pd.DataFrame,
        dates: pd.Series,
        positions: NDArray[np.int64],
        n: int,
    ) -> TemporalSplit:
        """Create a single temporal split."""
        train_end_idx = int(n * self.train_ratio)
        val_end_idx = int(n * (self.train_ratio + self.val_ratio))

        # Apply gap
        gap_offset = 0
        if self.gap_days > 0:
            # Find how many rows the gap represents
            train_end_date = dates.iloc[train_end_idx - 1]
            gap_end_date = train_end_date + timedelta(days=self.gap_days)
            gap_mask = dates > gap_end_date
            gap_offset = gap_mask[:val_end_idx].sum() - gap_mask[:train_end_idx].sum()

        train_start_pos = 0
        train_end_pos = train_end_idx
        val_start_pos = train_end_idx + gap_offset
        val_end_pos = val_end_idx + gap_offset
        test_start_pos = val_end_idx + gap_offset

        train_indices = positions[train_start_pos:train_end_pos]
        val_indices = positions[val_start_pos:val_end_pos]
        test_indices = positions[test_start_pos:n]

        return TemporalSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_start=dates.iloc[0],
            train_end=dates.iloc[train_end_idx - 1],
            val_start=dates.iloc[val_start_pos],
            val_end=dates.iloc[val_end_pos - 1],
            test_start=dates.iloc[test_start_pos],
            test_end=dates.iloc[-1],
        )

    def _rolling_splits(
        self,
        df: pd.DataFrame,
        dates: pd.Series,
        positions: NDArray[np.int64],
        n: int,
        n_splits: int,
    ) -> Iterator[TemporalSplit]:
        """Create rolling window splits for time-series cross-validation.

        Each split uses an expanding training window and fixed-size
        validation and test windows.
        """
        test_size = int(n * self.test_ratio)
        val_size = int(n * self.val_ratio)

        # Start from the end and work backwards
        for i in range(n_splits):
            test_end = n - (i * test_size)
            test_start = test_end - test_size
            val_end = test_start
            val_start = val_end - val_size
            train_end = val_start

            if val_start <= 0 or train_end <= 0:
                break

            train_indices = positions[:train_end]
            val_indices = positions[val_start:val_end]
            test_indices = positions[test_start:test_end]

            yield TemporalSplit(
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                train_start=dates.iloc[0],
                train_end=dates.iloc[train_end - 1],
                val_start=dates.iloc[val_start],
                val_end=dates.iloc[val_end - 1],
                test_start=dates.iloc[test_start],
                test_end=dates.iloc[test_end - 1],
            )


class OOFTimeAwareSplitter:
    """Time-aware OOF splitter for stacking and calibration.

    Generates K expanding-window folds where each fold's validation
    portion is strictly after its training portion. This prevents
    future-to-past leakage that shuffled StratifiedKFold introduces.

    Unlike WalkForwardSplitter (which produces full train/val/test
    triples), this splitter produces only train/val index pairs suitable
    for OOF prediction collection.

    Example (3 folds):
        Fold 1: Season 1-3: Train | Season 4: OOF
        Fold 2: Season 1-4: Train | Season 5: OOF
        Fold 3: Season 1-5: Train | Season 6: OOF
    """

    def __init__(
        self,
        n_folds: int = 5,
        season_column: str = "season_id",
        date_column: str = "match_date",
    ) -> None:
        self.n_folds = n_folds
        self.season_column = season_column
        self.date_column = date_column

    def split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate time-aware OOF folds.

        Args:
            df: DataFrame with season_id and match_date columns, sorted by date.

        Yields:
            (train_indices, oof_indices) tuples.
        """
        df, split_column = _with_split_season_column(
            df,
            self.season_column,
            self.date_column,
            "OOFTimeAwareSplitter",
        )

        df = df.sort_values(self.date_column)

        seasons = sorted(df[split_column].unique(), key=_season_sort_key)

        if len(seasons) < self.n_folds + 1:
            raise ValueError(
                f"Not enough seasons ({len(seasons)}) for {self.n_folds} OOF folds "
                f"(need at least {self.n_folds + 1})"
            )

        # Divide seasons into n_folds + 1 groups; use first groups cumulatively
        # for training and last group of each fold for OOF.
        # We take the last (n_folds + 1) seasons and split them.
        usable_seasons = seasons[-(self.n_folds + 1) :]

        for fold_idx in range(self.n_folds):
            # Training: all data before the OOF season
            oof_season = usable_seasons[fold_idx + 1]
            train_seasons = usable_seasons[: fold_idx + 1]

            train_mask = df[split_column].isin(train_seasons)
            oof_mask = df[split_column].isin([oof_season])

            train_indices = df.index[train_mask].to_numpy()
            oof_indices = df.index[oof_mask].to_numpy()

            if len(train_indices) == 0 or len(oof_indices) == 0:
                continue

            yield (train_indices, oof_indices)


class WalkForwardSplitter:
    """Walk-forward validation that respects season boundaries.

    Generates multiple train/val/test splits where each window
    shifts forward by one season. This provides honest out-of-sample
    performance estimates that are resistant to survivorship bias.

    Example (4 folds with train_seasons=6, val_seasons=1, test_seasons=1):
        Fold 1: Seasons 1-6: Train | Season 7: Val | Season 8: Test
        Fold 2: Seasons 2-7: Train | Season 8: Val | Season 9: Test
        ...
    """

    def __init__(
        self,
        train_seasons: int = 6,
        val_seasons: int = 1,
        test_seasons: int = 1,
        min_folds: int = 3,
        season_column: str = "season_id",
        date_column: str = "match_date",
    ) -> None:
        self.train_seasons = train_seasons
        self.val_seasons = val_seasons
        self.test_seasons = test_seasons
        self.min_folds = min_folds
        self.season_column = season_column
        self.date_column = date_column

    def split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[TemporalSplit]:
        """Generate walk-forward splits across seasons.

        Args:
            df: DataFrame with season_id and match_date columns

        Yields:
            TemporalSplit objects for each fold
        """
        df, split_column = _with_split_season_column(
            df,
            self.season_column,
            self.date_column,
            "WalkForwardSplitter",
        )

        df = df.sort_values(self.date_column)
        dates = pd.to_datetime(df[self.date_column])

        seasons = sorted(df[split_column].unique(), key=_season_sort_key)
        total_seasons_needed = self.train_seasons + self.val_seasons + self.test_seasons

        if len(seasons) < total_seasons_needed:
            raise ValueError(
                f"Not enough seasons ({len(seasons)}) for walk-forward split "
                f"(need {total_seasons_needed}: {self.train_seasons} train + "
                f"{self.val_seasons} val + {self.test_seasons} test)"
            )

        max_start = len(seasons) - total_seasons_needed + 1
        for start in range(max_start):
            train_sl = seasons[start : start + self.train_seasons]
            val_sl = seasons[
                start + self.train_seasons : start
                + self.train_seasons
                + self.val_seasons
            ]
            test_sl = seasons[
                start + self.train_seasons + self.val_seasons : start
                + self.train_seasons
                + self.val_seasons
                + self.test_seasons
            ]

            train_mask = df[split_column].isin(train_sl)
            val_mask = df[split_column].isin(val_sl)
            test_mask = df[split_column].isin(test_sl)

            train_indices = df.index[train_mask].to_numpy()
            val_indices = df.index[val_mask].to_numpy()
            test_indices = df.index[test_mask].to_numpy()

            if (
                len(train_indices) == 0
                or len(val_indices) == 0
                or len(test_indices) == 0
            ):
                continue

            yield TemporalSplit(
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                train_start=dates.loc[train_indices[0]],
                train_end=dates.loc[train_indices[-1]],
                val_start=dates.loc[val_indices[0]],
                val_end=dates.loc[val_indices[-1]],
                test_start=dates.loc[test_indices[0]],
                test_end=dates.loc[test_indices[-1]],
            )


class ExpandingWindowSplitter:
    """Expanding window splitter for time-series cross-validation.

    Uses an expanding training window (like sklearn's TimeSeriesSplit)
    but with proper validation and test sets.

    Example timeline:
    Split 1: [TRAIN-----][VAL][TEST]
    Split 2: [TRAIN----------][VAL][TEST]
    Split 3: [TRAIN---------------][VAL][TEST]
    """

    def __init__(
        self,
        min_train_size: int = 100,
        val_size: int = 50,
        test_size: int = 50,
        step_size: int = 50,
        date_column: str = "match_date",
    ) -> None:
        """Initialize expanding window splitter.

        Args:
            min_train_size: Minimum number of samples for initial training
            val_size: Number of samples for validation
            test_size: Number of samples for testing
            step_size: Number of samples to expand window each iteration
            date_column: Name of the date column
        """
        self.min_train_size = min_train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size
        self.date_column = date_column

    def split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[TemporalSplit]:
        """Generate expanding window splits.

        Args:
            df: DataFrame with match data

        Yields:
            TemporalSplit objects
        """
        df = df.sort_values(self.date_column)
        dates = pd.to_datetime(df[self.date_column])
        positions = df.index.to_numpy()
        n = len(df)

        train_end = self.min_train_size
        val_end = train_end + self.val_size
        test_end = val_end + self.test_size

        while test_end <= n:
            train_indices = positions[:train_end]
            val_indices = positions[train_end:val_end]
            test_indices = positions[val_end:test_end]

            yield TemporalSplit(
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                train_start=dates.iloc[0],
                train_end=dates.iloc[train_end - 1],
                val_start=dates.iloc[train_end],
                val_end=dates.iloc[val_end - 1],
                test_start=dates.iloc[val_end],
                test_end=dates.iloc[min(test_end, n) - 1],
            )

            # Expand window
            train_end += self.step_size
            val_end += self.step_size
            test_end += self.step_size


class SeasonAwareSplitter:
    """Split data by football seasons.

    Football has natural season boundaries that should be respected
    in data splitting. This splitter ensures:
    - Training on complete seasons only
    - Validation on complete seasons
    - Testing on the most recent season(s)

    Example:
        >>> splitter = SeasonAwareSplitter(
        ...     train_seasons=3,  # 2020/21, 2021/22, 2022/23
        ...     val_seasons=1,    # 2023/24
        ...     test_seasons=1,   # 2024/25
        ... )
    """

    def __init__(
        self,
        train_seasons: int = 3,
        val_seasons: int = 1,
        test_seasons: int = 1,
        season_column: str = "season_id",
        date_column: str = "match_date",
    ) -> None:
        """Initialize season-aware splitter.

        Args:
            train_seasons: Number of seasons for training
            val_seasons: Number of seasons for validation
            test_seasons: Number of seasons for testing
            season_column: Name of the season ID column
            date_column: Name of the date column
        """
        self.train_seasons = train_seasons
        self.val_seasons = val_seasons
        self.test_seasons = test_seasons
        self.season_column = season_column
        self.date_column = date_column

    def split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[TemporalSplit]:
        """Split data by seasons.

        Args:
            df: DataFrame with season_id column

        Yields:
            TemporalSplit objects
        """
        df, split_column = _with_split_season_column(
            df,
            self.season_column,
            self.date_column,
            "SeasonAwareSplitter",
        )

        df = df.sort_values(self.date_column)
        dates = pd.to_datetime(df[self.date_column])

        # Get unique seasons in chronological order
        seasons = df[split_column].unique()
        seasons = sorted(seasons, key=_season_sort_key)

        if len(seasons) < self.train_seasons + self.val_seasons + self.test_seasons:
            raise ValueError(
                f"Not enough seasons ({len(seasons)}) for requested split "
                f"({self.train_seasons} + {self.val_seasons} + {self.test_seasons})"
            )

        # Determine season groups
        test_season_start = len(seasons) - self.test_seasons
        val_season_start = test_season_start - self.val_seasons
        train_season_start = val_season_start - self.train_seasons

        train_seasons_list = seasons[train_season_start:val_season_start]
        val_seasons_list = seasons[val_season_start:test_season_start]
        test_seasons_list = seasons[test_season_start:]

        # Get indices
        train_mask = df[split_column].isin(train_seasons_list)
        val_mask = df[split_column].isin(val_seasons_list)
        test_mask = df[split_column].isin(test_seasons_list)

        train_indices = df.index[train_mask].to_numpy()
        val_indices = df.index[val_mask].to_numpy()
        test_indices = df.index[test_mask].to_numpy()

        yield TemporalSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_start=dates.loc[train_indices[0]]
            if len(train_indices) > 0
            else dates.iloc[0],
            train_end=dates.loc[train_indices[-1]]
            if len(train_indices) > 0
            else dates.iloc[0],
            val_start=dates.loc[val_indices[0]]
            if len(val_indices) > 0
            else dates.iloc[0],
            val_end=dates.loc[val_indices[-1]]
            if len(val_indices) > 0
            else dates.iloc[0],
            test_start=dates.loc[test_indices[0]]
            if len(test_indices) > 0
            else dates.iloc[0],
            test_end=dates.loc[test_indices[-1]]
            if len(test_indices) > 0
            else dates.iloc[0],
        )


def encode_targets(results: pd.Series | NDArray[Any]) -> NDArray[np.int64]:
    """Encode match results as integers.

    Args:
        results: Series/array with 'H', 'D', 'A' values

    Returns:
        Array of integers (0=H, 1=D, 2=A)
    """
    mapping = {"H": 0, "D": 1, "A": 2}

    if isinstance(results, pd.Series):
        return results.map(mapping).values
    else:
        return np.array([mapping.get(r, 1) for r in results])


def decode_targets(encoded: NDArray[np.int64]) -> list[str]:
    """Decode integer labels back to match results.

    Args:
        encoded: Array of integers (0=H, 1=D, 2=A)

    Returns:
        List of result strings
    """
    mapping = {0: "H", 1: "D", 2: "A"}
    return [mapping[int(e)] for e in encoded]


def get_class_weights(
    y: NDArray[np.int64],
    strength: float = 1.0,
) -> dict[int, float]:
    """Calculate class weights for imbalanced classification.

    Args:
        y: Array of encoded labels

    Returns:
        Dictionary mapping class to weight
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    strength = min(max(float(strength), 0.0), 1.0)
    weights = {}
    for cls, count in zip(unique, counts, strict=False):
        # Inverse frequency weighting
        balanced_weight = total / (len(unique) * count)
        weights[int(cls)] = 1.0 + ((balanced_weight - 1.0) * strength)

    return weights
