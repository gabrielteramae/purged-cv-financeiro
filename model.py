"""
Treino e avaliação do modelo de classificação (supera a mediana do setor
no horizonte definido?), usando o PurgedGroupTimeSeriesSplit como
validação cruzada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cv import PurgedGroupTimeSeriesSplit
from .data import ALL_FEATURES


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    auc: float


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_with_purged_cv(
    panel: pd.DataFrame,
    n_splits: int = 5,
    group_gap: int = 3,
    label_horizon: int = 5,
    features: list[str] | None = None,
) -> list[FoldResult]:
    """Roda a validação cruzada purgada e retorna a métrica (AUC) por fold."""
    features = features or ALL_FEATURES

    X = panel[features].to_numpy()
    y = panel["label"].to_numpy()
    groups = panel["date"].to_numpy()  # cada grupo = um pregão

    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=n_splits,
        group_gap=group_gap,
        label_horizon=label_horizon,
    )

    results: list[FoldResult] = []
    for fold_i, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        pipeline = build_pipeline()
        pipeline.fit(X[train_idx], y[train_idx])
        proba = pipeline.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], proba)

        train_dates = panel["date"].to_numpy()[train_idx]
        test_dates = panel["date"].to_numpy()[test_idx]

        results.append(
            FoldResult(
                fold=fold_i,
                n_train=len(train_idx),
                n_test=len(test_idx),
                train_start=str(pd.Timestamp(train_dates.min()).date()),
                train_end=str(pd.Timestamp(train_dates.max()).date()),
                test_start=str(pd.Timestamp(test_dates.min()).date()),
                test_end=str(pd.Timestamp(test_dates.max()).date()),
                auc=float(auc),
            )
        )

    return results


def summarize(results: list[FoldResult]) -> dict:
    aucs = np.array([r.auc for r in results])
    return {
        "n_folds": len(results),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std()),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
    }
