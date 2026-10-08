# Purged CV Financeiro — validação sem vazamento temporal

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-7.4+-0A9EDC?logo=pytest&logoColor=white)

O repositório compara duas validações do mesmo classificador num painel sintético (ativo × pregão). `PurgedGroupTimeSeriesSplit` mantém cada data inteira de um lado só, corta do treino o horizonte do rótulo e ainda aplica um embargo. Um `KFold` aleatório faz o contrário, de propósito. Os preços não são de mercado e o AUC não é sinal para operar.

| Protocolo | O que faz com o tempo | Papel aqui |
| --- | --- | --- |
| `PurgedGroupTimeSeriesSplit` | grupo = data; purge de `label_horizon`; embargo de `group_gap`; janela expansível ou limitada por `max_train_group_size` | validação que `main.py` trata como correta |
| `KFold(shuffle=True)` | embaralha linhas; a mesma data e a janela do `fwd_return` podem cair nos dois lados | contraste para mostrar o AUC inflado |

## Stack

- scikit-learn >= 1.3 (`BaseCrossValidator`, `GradientBoostingClassifier`, `StandardScaler`, `KFold`, AUC)
- pandas >= 2.0 e NumPy >= 1.26, para o painel sintético
- pytest >= 7.4
- o repositório não fixa a versão do Python

## Estrutura

```
.
├── cv.py              # PurgedGroupTimeSeriesSplit
├── data.py            # painel sintético e listas de features
├── model.py           # pipeline e AUC por fold
├── main.py            # purged contra KFold
├── test_cv.py         # asserts do splitter
└── requirements.txt
```

O painel padrão tem 60 ativos, 750 pregões a partir de 2021-01-04, 6 setores e `label_horizon` 5. Fundamentos (`value_score`, `quality_score`, `growth_score`, `earnings_surprise`) atualizam a cada 21 pregões. Microestrutura (`bid_ask_spread`, `order_imbalance`, `rel_volume`, `realized_vol`) muda todo dia. O rótulo é 1 quando o retorno futuro supera a mediana do setor naquela data. O classificador é `GradientBoostingClassifier` (200 árvores, `max_depth` 3, `learning_rate` 0.05, `subsample` 0.8) depois de `StandardScaler`. Em `main.py`: 5 folds e `group_gap` 3.

## Como rodar

```bash
git clone https://github.com/gabrielteramae/purged-cv-financeiro.git
cd purged-cv-financeiro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Os módulos não rodam como estão na árvore. `main.py` e `model.py` usam import relativo (`from .data`, `from .cv`) e o docstring pede `python -m src.main`. `test_cv.py` importa `src.cv`. Não existe pasta `src/` nem `__init__.py`. Por isso `python main.py`, `python -m src.main` e `pytest` falham até os quatro módulos irem para um pacote `src` ou os imports passarem a ser os da raiz.

## Testes realizados

`test_cv.py` tem seis testes, lidos no arquivo, ainda sem como importar `src.cv` neste layout:

- nenhum grupo aparece em treino e teste no mesmo fold
- o maior grupo de treino é sempre anterior ao menor grupo de teste
- o buraco antes do teste é pelo menos `label_horizon + group_gap`
- `max_train_group_size=10` limita os grupos de treino
- `groups=None` levanta `ValueError`
- `get_n_splits()` devolve o `n_splits` pedido

---

© 2026 Gabriel Teramae Chan
