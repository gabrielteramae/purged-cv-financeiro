"""
Geração de um painel sintético (ativo x data) com dois blocos de features:

1. Fundamentos econômicos — atualizam pouco (trimestral/mensal) e
   carregam sinal de médio prazo: value, quality, growth, earnings
   surprise.
2. Microestrutura — atualizam a cada pregão e carregam sinal de curto
   prazo: spread bid-ask, desequilíbrio de ordens (order imbalance),
   volume relativo, volatilidade realizada intradiária.

O rótulo (target) é binário: 1 se o ativo supera a mediana de retorno do
"setor" (grupo) no horizonte definido, 0 caso contrário — um desenho
comum em modelos de ranking cross-sectional, em que o Purged Group
TimeSeriesSplit é especialmente importante (mesma data = mesmo grupo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_panel(
    n_assets: int = 60,
    n_days: int = 750,
    label_horizon: int = 5,
    n_sectors: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera um DataFrame long-format: uma linha por (ativo, data).

    Returns
    -------
    pd.DataFrame com colunas:
        date, asset_id, sector,
        value_score, quality_score, growth_score, earnings_surprise,
        bid_ask_spread, order_imbalance, rel_volume, realized_vol,
        fwd_return, label
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2021-01-04", periods=n_days)
    asset_ids = np.array([f"AST{idx:03d}" for idx in range(n_assets)])
    sector_of_asset = rng.integers(0, n_sectors, size=n_assets)

    # --- fatores latentes por ativo (drivam sinal real, não só ruído) ---
    true_value_alpha = rng.normal(0, 1, size=n_assets)
    true_quality_alpha = rng.normal(0, 1, size=n_assets)

    rows = []
    # fundamentos mudam pouco: atualiza a cada ~21 pregões (mensal)
    fundamentals_refresh = 21

    value_score = rng.normal(0, 1, size=n_assets)
    quality_score = rng.normal(0, 1, size=n_assets)
    growth_score = rng.normal(0, 1, size=n_assets)
    earnings_surprise = rng.normal(0, 1, size=n_assets)

    # ruído idiossincrático de preço acumulado por ativo (passeio aleatório)
    log_price = rng.normal(0, 0.01, size=n_assets).cumsum() * 0 + 100.0

    for t, date in enumerate(dates):
        if t % fundamentals_refresh == 0 and t > 0:
            value_score = 0.85 * value_score + rng.normal(0, 0.5, size=n_assets)
            quality_score = 0.85 * quality_score + rng.normal(0, 0.5, size=n_assets)
            growth_score = 0.85 * growth_score + rng.normal(0, 0.5, size=n_assets)
            earnings_surprise = rng.normal(0, 1, size=n_assets)

        bid_ask_spread = np.abs(rng.normal(0.05, 0.02, size=n_assets))
        order_imbalance = rng.normal(0, 1, size=n_assets)
        rel_volume = np.abs(rng.normal(1.0, 0.4, size=n_assets))
        realized_vol = np.abs(rng.normal(0.015, 0.006, size=n_assets))

        # sinal real (o que o modelo *deveria* aprender a captar):
        # value + quality de médio prazo, e order_imbalance de curto prazo
        signal = (
            0.35 * true_value_alpha
            + 0.25 * true_quality_alpha
            + 0.15 * order_imbalance
            + 0.10 * earnings_surprise
        )
        daily_return = 0.0003 * signal + rng.normal(0, 0.012, size=n_assets)
        log_price = log_price * np.exp(daily_return)

        rows.append(
            pd.DataFrame(
                {
                    "date": date,
                    "asset_id": asset_ids,
                    "sector": sector_of_asset,
                    "value_score": value_score,
                    "quality_score": quality_score,
                    "growth_score": growth_score,
                    "earnings_surprise": earnings_surprise,
                    "bid_ask_spread": bid_ask_spread,
                    "order_imbalance": order_imbalance,
                    "rel_volume": rel_volume,
                    "realized_vol": realized_vol,
                    "price": log_price,
                }
            )
        )

    panel = pd.concat(rows, ignore_index=True)
    panel.sort_values(["asset_id", "date"], inplace=True)

    # retorno futuro (forward return) no horizonte definido, por ativo
    panel["fwd_return"] = (
        panel.groupby("asset_id")["price"]
        .transform(lambda s: s.shift(-label_horizon) / s - 1.0)
    )

    # label cross-sectional: supera a mediana do SETOR na mesma data?
    sector_median = panel.groupby(["date", "sector"])["fwd_return"].transform("median")
    panel["label"] = (panel["fwd_return"] > sector_median).astype("Int64")

    # remove as últimas `label_horizon` datas de cada ativo (sem fwd_return)
    panel = panel.dropna(subset=["fwd_return", "label"]).reset_index(drop=True)
    panel["label"] = panel["label"].astype(int)

    return panel


FUNDAMENTAL_FEATURES = [
    "value_score",
    "quality_score",
    "growth_score",
    "earnings_surprise",
]

MICROSTRUCTURE_FEATURES = [
    "bid_ask_spread",
    "order_imbalance",
    "rel_volume",
    "realized_vol",
]

ALL_FEATURES = FUNDAMENTAL_FEATURES + MICROSTRUCTURE_FEATURES
