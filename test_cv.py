import numpy as np
import pytest

from src.cv import PurgedGroupTimeSeriesSplit


def make_toy_data(n_groups=60, rows_per_group=3):
    """Cada grupo = uma 'data'; várias linhas por grupo = vários ativos."""
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    X = np.random.default_rng(0).normal(size=(len(groups), 2))
    y = np.random.default_rng(1).integers(0, 2, size=len(groups))
    return X, y, groups


def test_no_group_is_split_between_train_and_test():
    X, y, groups = make_toy_data()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=4, group_gap=2, label_horizon=1)

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups), (
            "Um mesmo grupo (data) apareceu em treino E teste no mesmo fold."
        )


def test_train_is_always_chronologically_before_test():
    X, y, groups = make_toy_data()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=4, group_gap=2, label_horizon=1)

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        assert groups[train_idx].max() < groups[test_idx].min()


def test_purge_and_embargo_leave_a_real_gap_before_test():
    X, y, groups = make_toy_data()
    label_horizon = 4
    group_gap = 3
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=4, group_gap=group_gap, label_horizon=label_horizon
    )

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        gap = groups[test_idx].min() - groups[train_idx].max()
        # o gap real deve ser pelo menos label_horizon + group_gap
        assert gap >= label_horizon + group_gap


def test_rolling_window_limits_train_size():
    X, y, groups = make_toy_data(n_groups=80, rows_per_group=2)
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=5, group_gap=1, label_horizon=1, max_train_group_size=10
    )

    for train_idx, _test_idx in splitter.split(X, y, groups=groups):
        n_train_groups = len(set(groups[train_idx]))
        assert n_train_groups <= 10


def test_raises_without_groups():
    X, y, groups = make_toy_data()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=3)
    with pytest.raises(ValueError):
        list(splitter.split(X, y, groups=None))


def test_get_n_splits():
    splitter = PurgedGroupTimeSeriesSplit(n_splits=6)
    assert splitter.get_n_splits() == 6
