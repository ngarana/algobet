"""Feature store for caching and retrieving computed features.

The feature store provides a persistent cache for computed features,
supporting versioning, time-travel queries, and efficient bulk operations.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import and_, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from algobet.models import Match, ModelFeature


class FeatureStore:
    """Persistent storage for computed features.

    Provides caching, versioning, and retrieval of features computed
    for matches. Supports:
    - Schema versioning for feature evolution
    - Bulk operations for efficiency
    - Time-travel queries for historical features
    - Automatic cache invalidation

    Usage:
        store = FeatureStore(session, schema_version="v1.0")

        # Store features
        store.store_features(match_id, {"home_form": 1.5, "away_form": 2.0})

        # Retrieve features
        features = store.get_features(match_id)

        # Bulk operations
        store.store_bulk(features_list)
        df = store.get_features_for_matches([1, 2, 3])
    """

    def __init__(
        self,
        session: Session,
        schema_version: str = "v1.0",
    ) -> None:
        """Initialize feature store.

        Args:
            session: SQLAlchemy database session
            schema_version: Version of feature schema being used
        """
        self.session = session
        self.schema_version = schema_version

    def store_features(
        self,
        match_id: int,
        features: dict[str, Any],
    ) -> ModelFeature:
        """Store features for a single match.

        Uses upsert to handle existing features for the same
        match and schema version.

        Args:
            match_id: ID of the match
            features: Dictionary of feature name to value

        Returns:
            Created or updated ModelFeature record
        """
        # Use PostgreSQL upsert
        stmt = insert(ModelFeature).values(
            match_id=match_id,
            feature_schema_version=self.schema_version,
            features=features,
            computed_at=datetime.now(),
        )

        # On conflict, update features
        stmt = stmt.on_conflict_do_update(
            constraint="uq_match_features_schema",
            set_={
                "features": stmt.excluded.features,
                "computed_at": stmt.excluded.computed_at,
            },
        ).returning(ModelFeature)

        result = self.session.execute(stmt)
        return result.scalar_one()

    def store_bulk(
        self,
        features_list: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Store features for multiple matches efficiently.

        Args:
            features_list: List of dicts with 'match_id' and feature values
            batch_size: Number of records to insert per batch

        Returns:
            Number of records stored
        """
        if not features_list:
            return 0

        stored = 0

        for i in range(0, len(features_list), batch_size):
            batch = features_list[i : i + batch_size]

            records = [
                {
                    "match_id": item["match_id"],
                    "feature_schema_version": self.schema_version,
                    "features": {k: v for k, v in item.items() if k != "match_id"},
                    "computed_at": datetime.now(),
                }
                for item in batch
            ]

            # Use upsert for each batch
            for record in records:
                stmt = insert(ModelFeature).values(**record)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_match_features_schema",
                    set_={
                        "features": stmt.excluded.features,
                        "computed_at": stmt.excluded.computed_at,
                    },
                )
                self.session.execute(stmt)

            stored += len(batch)

        return stored

    def get_features(
        self,
        match_id: int,
        schema_version: str | None = None,
    ) -> dict[str, Any] | None:
        """Get features for a single match.

        Args:
            match_id: ID of the match
            schema_version: Optional specific schema version (default: current)

        Returns:
            Feature dictionary or None if not found
        """
        schema = schema_version or self.schema_version

        stmt = select(ModelFeature).where(
            and_(
                ModelFeature.match_id == match_id,
                ModelFeature.feature_schema_version == schema,
            )
        )

        result = self.session.execute(stmt)
        feature_record = result.scalar_one_or_none()

        if feature_record is None:
            return None

        return feature_record.features

    def get_features_for_matches(
        self,
        match_ids: list[int],
        schema_version: str | None = None,
    ) -> pd.DataFrame:
        """Get features for multiple matches as DataFrame.

        Args:
            match_ids: List of match IDs
            schema_version: Optional specific schema version

        Returns:
            DataFrame with match_id index and feature columns
        """
        if not match_ids:
            return pd.DataFrame()

        schema = schema_version or self.schema_version

        stmt = select(ModelFeature).where(
            and_(
                ModelFeature.match_id.in_(match_ids),
                ModelFeature.feature_schema_version == schema,
            )
        )

        result = self.session.execute(stmt)
        records = list(result.scalars().all())

        if not records:
            return pd.DataFrame(index=match_ids)

        # Convert to DataFrame
        rows = []
        for record in records:
            row = {"match_id": record.match_id}
            row.update(record.features)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.set_index("match_id")

    def get_missing_match_ids(
        self,
        match_ids: list[int],
    ) -> list[int]:
        """Find match IDs that don't have cached features.

        Args:
            match_ids: List of match IDs to check

        Returns:
            List of match IDs without cached features
        """
        if not match_ids:
            return []

        stmt = select(ModelFeature.match_id).where(
            and_(
                ModelFeature.match_id.in_(match_ids),
                ModelFeature.feature_schema_version == self.schema_version,
            )
        )

        result = self.session.execute(stmt)
        existing_ids = {row[0] for row in result.fetchall()}

        return [mid for mid in match_ids if mid not in existing_ids]

    def get_features_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        schema_version: str | None = None,
    ) -> pd.DataFrame:
        """Get features for all matches in a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            schema_version: Optional specific schema version

        Returns:
            DataFrame with match features
        """
        schema = schema_version or self.schema_version

        stmt = (
            select(ModelFeature, Match.match_date)
            .join(Match, ModelFeature.match_id == Match.id)
            .where(
                and_(
                    ModelFeature.feature_schema_version == schema,
                    Match.match_date >= start_date,
                    Match.match_date <= end_date,
                )
            )
            .order_by(Match.match_date)
        )

        result = self.session.execute(stmt)
        rows = []

        for feature_record, match_date in result:
            row = {
                "match_id": feature_record.match_id,
                "match_date": match_date,
            }
            row.update(feature_record.features)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def delete_features(
        self,
        match_id: int,
        schema_version: str | None = None,
    ) -> bool:
        """Delete features for a match.

        Args:
            match_id: ID of the match
            schema_version: Optional specific schema version

        Returns:
            True if features were deleted
        """
        schema = schema_version or self.schema_version

        stmt = delete(ModelFeature).where(
            and_(
                ModelFeature.match_id == match_id,
                ModelFeature.feature_schema_version == schema,
            )
        )

        result = self.session.execute(stmt)
        return result.rowcount > 0

    def clear_schema(self, schema_version: str | None = None) -> int:
        """Clear all features for a schema version.

        Use with caution - this deletes all cached features.

        Args:
            schema_version: Schema version to clear (default: current)

        Returns:
            Number of records deleted
        """
        schema = schema_version or self.schema_version

        stmt = delete(ModelFeature).where(
            ModelFeature.feature_schema_version == schema
        )

        result = self.session.execute(stmt)
        return result.rowcount

    def get_schema_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics about stored features by schema version.

        Returns:
            Dictionary mapping schema version to stats
        """
        stmt = select(ModelFeature)

        result = self.session.execute(stmt)
        records = list(result.scalars().all())

        stats: dict[str, dict[str, Any]] = {}

        for record in records:
            schema = record.feature_schema_version

            if schema not in stats:
                stats[schema] = {
                    "count": 0,
                    "feature_names": set(),
                    "oldest_computed": record.computed_at,
                    "newest_computed": record.computed_at,
                }

            stats[schema]["count"] += 1
            stats[schema]["feature_names"].update(record.features.keys())

            if record.computed_at < stats[schema]["oldest_computed"]:
                stats[schema]["oldest_computed"] = record.computed_at
            if record.computed_at > stats[schema]["newest_computed"]:
                stats[schema]["newest_computed"] = record.computed_at

        # Convert sets to lists for JSON compatibility
        for schema_data in stats.values():
            schema_data["feature_names"] = sorted(schema_data["feature_names"])

        return stats

    def get_computed_at(
        self,
        match_id: int,
        schema_version: str | None = None,
    ) -> datetime | None:
        """Get when features were computed for a match.

        Args:
            match_id: ID of the match
            schema_version: Optional specific schema version

        Returns:
            Datetime when features were computed or None
        """
        schema = schema_version or self.schema_version

        stmt = select(ModelFeature.computed_at).where(
            and_(
                ModelFeature.match_id == match_id,
                ModelFeature.feature_schema_version == schema,
            )
        )

        result = self.session.execute(stmt)
        computed_at = result.scalar_one_or_none()

        return computed_at

    def is_cached(
        self,
        match_id: int,
        schema_version: str | None = None,
    ) -> bool:
        """Check if features are cached for a match.

        Args:
            match_id: ID of the match
            schema_version: Optional specific schema version

        Returns:
            True if features are cached
        """
        return self.get_features(match_id, schema_version) is not None

    def bulk_is_cached(
        self,
        match_ids: list[int],
        schema_version: str | None = None,
    ) -> dict[int, bool]:
        """Check if features are cached for multiple matches.

        Args:
            match_ids: List of match IDs
            schema_version: Optional specific schema version

        Returns:
            Dictionary mapping match_id to cached status
        """
        schema = schema_version or self.schema_version

        stmt = select(ModelFeature.match_id).where(
            and_(
                ModelFeature.match_id.in_(match_ids),
                ModelFeature.feature_schema_version == schema,
            )
        )

        result = self.session.execute(stmt)
        cached_ids = {row[0] for row in result.fetchall()}

        return {mid: mid in cached_ids for mid in match_ids}


def features_to_store_format(
    features_df: pd.DataFrame,
    schema_version: str = "v1.0",
) -> list[dict[str, Any]]:
    """Convert feature DataFrame to storage format.

    Args:
        features_df: DataFrame with match_id index and feature columns
        schema_version: Schema version to use

    Returns:
        List of dicts suitable for store_bulk()
    """
    records = []

    for match_id, row in features_df.iterrows():
        record = {"match_id": int(match_id)}
        record.update(row.to_dict())
        records.append(record)

    return records
