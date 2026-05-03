"""Unit tests for training classifiers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from algobet.predictions.training.classifiers import ModelConfig, XGBoostPredictor


def test_xgboost_sycl_trains_one_vs_rest_binary_boosters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SYCL training should avoid the crashing multiclass path."""
    train_calls: list[dict[str, object]] = []

    def fake_dmatrix(
        data: np.ndarray,
        label: np.ndarray | None = None,
        weight: np.ndarray | None = None,
    ) -> dict[str, object]:
        return {
            "rows": len(data),
            "label": None if label is None else tuple(int(v) for v in label),
            "has_weight": weight is not None,
        }

    def fake_train(
        params: dict[str, object],
        dtrain: object,
        *,
        num_boost_round: int,
        evals: object,
        early_stopping_rounds: object,
        verbose_eval: bool,
    ) -> MagicMock:
        train_calls.append(
            {
                "params": params,
                "dtrain": dtrain,
                "evals": evals,
                "early_stopping_rounds": early_stopping_rounds,
                "num_boost_round": num_boost_round,
                "verbose_eval": verbose_eval,
            }
        )
        return MagicMock()

    monkeypatch.setattr(
        "algobet.predictions.training.classifiers.xgb.DMatrix",
        fake_dmatrix,
    )
    monkeypatch.setattr(
        "algobet.predictions.training.classifiers.xgb.train",
        fake_train,
    )

    X = np.random.randn(24, 6).astype(np.float64)
    y = np.array([0, 1, 2] * 8, dtype=np.int64)

    predictor = XGBoostPredictor(
        ModelConfig(
            model_type="xgboost",
            hyperparameters={
                "device": "sycl:gpu:0",
                "tree_method": "hist",
                "n_estimators": 12,
            },
        )
    )

    with pytest.warns(RuntimeWarning, match="one-vs-rest binary boosters"):
        predictor.fit(X[:18], y[:18], X[18:], y[18:])

    assert len(train_calls) == 3
    for call in train_calls:
        assert call["evals"] is None
        assert call["early_stopping_rounds"] is None
        assert call["num_boost_round"] == 12
        assert call["params"] == {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "device": "sycl:gpu:0",
            "tree_method": "hist",
        }
        labels = call["dtrain"]["label"]
        assert labels is not None
        assert set(labels).issubset({0, 1})


def test_xgboost_cpu_keeps_eval_sets_and_early_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-SYCL training should keep the normal eval/early stopping flow."""
    train_calls: dict[str, object] = {}

    def fake_dmatrix(
        data: np.ndarray,
        label: np.ndarray | None = None,
        weight: np.ndarray | None = None,
    ) -> dict[str, object]:
        return {
            "rows": len(data),
            "label": None if label is None else tuple(int(v) for v in label),
            "has_weight": weight is not None,
        }

    def fake_train(
        params: dict[str, object],
        dtrain: object,
        *,
        num_boost_round: int,
        evals: object,
        early_stopping_rounds: object,
        verbose_eval: bool,
    ) -> MagicMock:
        train_calls["params"] = params
        train_calls["evals"] = evals
        train_calls["early_stopping_rounds"] = early_stopping_rounds
        train_calls["num_boost_round"] = num_boost_round
        train_calls["verbose_eval"] = verbose_eval
        return MagicMock()

    monkeypatch.setattr(
        "algobet.predictions.training.classifiers.xgb.DMatrix",
        fake_dmatrix,
    )
    monkeypatch.setattr(
        "algobet.predictions.training.classifiers.xgb.train",
        fake_train,
    )

    X = np.random.randn(24, 6).astype(np.float64)
    y = np.array([0, 1, 2] * 8, dtype=np.int64)

    predictor = XGBoostPredictor(
        ModelConfig(
            model_type="xgboost",
            hyperparameters={
                "device": "cpu",
                "tree_method": "hist",
                "n_estimators": 12,
            },
        )
    )

    predictor.fit(X[:18], y[:18], X[18:], y[18:])

    assert train_calls["evals"] == [
        (
            {
                "rows": 18,
                "label": tuple(int(v) for v in y[:18]),
                "has_weight": False,
            },
            "train",
        ),
        (
            {
                "rows": 6,
                "label": tuple(int(v) for v in y[18:]),
                "has_weight": False,
            },
            "val",
        ),
    ]
    assert train_calls["early_stopping_rounds"] == 50
    assert train_calls["num_boost_round"] == 12


def test_xgboost_sycl_predict_proba_normalizes_one_vs_rest_outputs() -> None:
    """One-vs-rest SYCL boosters should return normalized multiclass probabilities."""

    class DummyBooster:
        def __init__(self, predictions: list[float]) -> None:
            self._predictions = np.array(predictions, dtype=np.float64)

        def predict(self, dmatrix: object) -> np.ndarray:
            return self._predictions

    predictor = XGBoostPredictor(
        ModelConfig(
            model_type="xgboost",
            hyperparameters={"device": "sycl:gpu:0", "tree_method": "hist"},
        )
    )
    predictor._model = [
        DummyBooster([0.2, 0.1]),
        DummyBooster([0.3, 0.2]),
        DummyBooster([0.5, 0.7]),
    ]
    predictor._is_fitted = True

    probas = predictor.predict_proba(np.zeros((2, 4), dtype=np.float64))

    assert probas.shape == (2, 3)
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(2))
