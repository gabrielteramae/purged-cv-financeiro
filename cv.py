"""
Validação cruzada purgada para séries temporais financeiras (Purged Group
TimeSeriesSplit), no espírito de López de Prado (Advances in Financial
Machine Learning, cap. 7).

Problema que isso resolve
--------------------------
Num K-Fold comum (ou até num TimeSeriesSplit "ingênuo"), observações de
treino podem "vazar" informação do futuro para o teste quando:

1. Os rótulos (labels) usam uma janela futura (ex.: retorno nos próximos
   N dias) — então uma amostra de treino "vê" um pedaço de tempo que
   também aparece no período de teste.
2. Existe autocorrelação/serial dependence entre observações vizinhas no
   tempo, então treino e teste "colados" inflam artificialmente a
   performance.

Este splitter resolve isso com duas técnicas combinadas:

- **Agrupamento (group)**: cada grupo (normalmente uma data de pregão)
  fica inteiro de um lado só (treino OU teste), nunca dividido.
- **Purge**: remove do treino qualquer grupo cuja janela de rótulo
  (t_start, t_end) se sobreponha à janela de teste.
- **Embargo**: remove também um pequeno intervalo de grupos logo APÓS
  o fim do teste, pra evitar vazamento por autocorrelação residual.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """K-Fold expansível no tempo, com purge e embargo, respeitando grupos.

    Parameters
    ----------
    n_splits : int
        Número de folds de teste (sempre nos blocos finais da série).
    group_gap : int
        Número de grupos de "embargo" removidos do treino logo após cada
        janela de teste.
    max_train_group_size : int | None
        Se definido, limita o treino a uma janela deslizante (rolling)
        com esse tamanho em número de grupos, em vez de expandir
        indefinidamente.
    label_horizon : int
        Quantos grupos à frente a label de uma observação "enxerga"
        (ex.: se a label é o retorno nos próximos 5 dias e cada grupo é
        1 dia, use label_horizon=5). Usado para purgar corretamente
        observações de treino cuja janela de rótulo entra no teste.
    """

    def __init__(
        self,
        n_splits: int = 5,
        group_gap: int = 2,
        max_train_group_size: int | None = None,
        label_horizon: int = 1,
    ):
        if n_splits < 2:
            raise ValueError("n_splits precisa ser >= 2")
        self.n_splits = n_splits
        self.group_gap = group_gap
        self.max_train_group_size = max_train_group_size
        self.label_horizon = label_horizon

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError(
                "groups é obrigatório (ex.: a data de pregão de cada linha)."
            )

        groups = np.asarray(groups)
        unique_groups, group_index = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits >= n_groups:
            raise ValueError(
                f"n_splits ({self.n_splits}) precisa ser menor que o número "
                f"de grupos únicos ({n_groups})."
            )

        # divide os grupos (em ordem cronológica) em n_splits+1 blocos:
        # os primeiros blocos formam a base de treino que vai crescendo,
        # e cada um dos últimos n_splits blocos vira um fold de teste.
        test_block_size = n_groups // (self.n_splits + 1)
        if test_block_size < 1:
            raise ValueError("Grupos insuficientes para o número de splits pedido.")

        for split_i in range(self.n_splits):
            test_start = (split_i + 1) * test_block_size
            test_end = (
                n_groups
                if split_i == self.n_splits - 1
                else test_start + test_block_size
            )

            test_group_ids = np.arange(test_start, test_end)
            train_group_ids = np.arange(0, test_start)

            # --- purge + embargo: remove do fim do treino (a) qualquer
            # grupo cuja janela de rótulo (group_id .. group_id +
            # label_horizon) encoste no início do teste, e (b) uma folga
            # extra de `group_gap` grupos, como margem de segurança contra
            # autocorrelação residual entre observações vizinhas ---
            purge_cutoff = test_start - self.label_horizon - self.group_gap
            train_group_ids = train_group_ids[train_group_ids < purge_cutoff]

            # --- rolling window opcional, em vez de expanding window ---
            if self.max_train_group_size is not None and len(train_group_ids) > 0:
                min_allowed = train_group_ids.max() - self.max_train_group_size + 1
                train_group_ids = train_group_ids[train_group_ids >= min_allowed]

            if len(train_group_ids) == 0:
                continue

            train_mask = np.isin(group_index, train_group_ids)
            test_mask = np.isin(group_index, test_group_ids)

            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)
