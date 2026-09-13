"""
Pipeline fim-a-fim:

1. Gera o painel sintético (fundamentos + microestrutura).
2. Avalia o modelo com PurgedGroupTimeSeriesSplit (validação correta).
3. Avalia o MESMO modelo com um KFold aleatório comum, sem purge/grupo
   (validação "ingênua"/com vazamento), pra mostrar o quanto isso infla
   a métrica de forma artificial.

Rode com:  python -m src.main
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from .data import ALL_FEATURES, generate_panel
from .model import build_pipeline, evaluate_with_purged_cv, summarize


def naive_kfold_auc(panel, n_splits: int = 5, seed: int = 42) -> list[float]:
    """K-Fold aleatório comum — embaralha linhas, ignora tempo e grupo.

    Isso deliberadamente comete os dois erros que o PurgedGroupTimeSeriesSplit
    evita: (a) observações da MESMA data podem cair uma em treino e outra em
    teste, e (b) a janela de rótulo (fwd_return) do treino pode se sobrepor
    à do teste. Serve só de contraste pedagógico.
    """
    X = panel[ALL_FEATURES].to_numpy()
    y = panel["label"].to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in kf.split(X):
        pipeline = build_pipeline()
        pipeline.fit(X[train_idx], y[train_idx])
        proba = pipeline.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return aucs


def main():
    label_horizon = 5
    panel = generate_panel(n_assets=60, n_days=750, label_horizon=label_horizon)

    print(f"Painel gerado: {len(panel):,} linhas "
          f"({panel['asset_id'].nunique()} ativos x {panel['date'].nunique()} pregões)")
    print(f"Taxa de positivos (label=1): {panel['label'].mean():.3f}\n")

    print("=" * 72)
    print("Validação CORRETA — PurgedGroupTimeSeriesSplit")
    print("=" * 72)
    results = evaluate_with_purged_cv(
        panel, n_splits=5, group_gap=3, label_horizon=label_horizon
    )
    for r in results:
        print(
            f"Fold {r.fold}: treino {r.train_start} -> {r.train_end} "
            f"({r.n_train:,} linhas)  |  teste {r.test_start} -> {r.test_end} "
            f"({r.n_test:,} linhas)  |  AUC = {r.auc:.4f}"
        )
    purged_summary = summarize(results)
    print(f"\nAUC médio (purged): {purged_summary['auc_mean']:.4f} "
          f"+/- {purged_summary['auc_std']:.4f}")

    print("\n" + "=" * 72)
    print("Validação INGÊNUA — KFold aleatório (com vazamento de grupo/tempo)")
    print("=" * 72)
    naive_aucs = naive_kfold_auc(panel, n_splits=5)
    print(f"AUC por fold: {[round(a, 4) for a in naive_aucs]}")
    print(f"AUC médio (naive): {np.mean(naive_aucs):.4f} +/- {np.std(naive_aucs):.4f}")

    print("\n" + "=" * 72)
    diff = np.mean(naive_aucs) - purged_summary["auc_mean"]
    print(
        f"Diferença (naive - purged) = {diff:+.4f} -> "
        f"{'vazamento infla a métrica' if diff > 0 else 'sem diferença relevante'}"
    )


if __name__ == "__main__":
    main()
