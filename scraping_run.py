from sqlalchemy import select
from sqlalchemy.orm import joinedload

from algobet.infrastructure.database import session_scope
from algobet.models import BacktestHistory, Match, ModelVersion
from algobet.services.prediction_service import PredictionService

version = "xgboost_20260508_035804"


def run_stats():
    with session_scope() as session:
        model = session.execute(
            select(ModelVersion).where(ModelVersion.version == version)
        ).scalar_one()
        print("MODEL_METRICS")
        for k in sorted(model.metrics.keys()):
            if k.startswith(("test_", "val_")):
                print(f"{k}={model.metrics[k]}")
        print("BACKTEST_HISTORY")
        rows = (
            session.execute(
                select(BacktestHistory)
                .join(ModelVersion)
                .where(ModelVersion.version == version)
            )
            .scalars()
            .all()
        )
        print("count", len(rows))
        for row in rows:
            print(
                {
                    "evaluated_at": str(row.evaluated_at),
                    "num_samples": row.num_samples,
                    "date_range_start": row.date_range_start,
                    "date_range_end": row.date_range_end,
                    "accuracy": row.accuracy,
                    "log_loss": row.log_loss,
                    "f1_macro": row.f1_macro,
                    "roi_percent": row.roi_percent,
                    "win_rate": row.win_rate,
                }
            )


def run_prediction():
    match_id = 19589
    with session_scope() as session:
        service = PredictionService(session)
        model, _ = service.load_model(version)
        match = session.execute(
            select(Match)
            .options(joinedload(Match.home_team), joinedload(Match.away_team))
            .where(Match.id == match_id)
        ).scalar_one()
        result = service.predict_match(match, model_version=version)
        actual = (
            "HOME"
            if match.home_score > match.away_score
            else "AWAY"
            if match.home_score < match.away_score
            else "DRAW"
        )
        print(
            {
                "predicted_outcome": result.predicted_outcome,
                "confidence": result.confidence,
                "prob_home": result.prob_home,
                "prob_draw": result.prob_draw,
                "prob_away": result.prob_away,
                "actual_outcome": actual,
                "actual_score": f"{match.home_score}-{match.away_score}",
            }
        )


if __name__ == "__main__":
    run_stats()
    run_prediction()
