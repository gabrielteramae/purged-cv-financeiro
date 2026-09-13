# 📈 Purged CV Financeiro

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0-150458?style=flat&logo=pandas&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-7.4-0A9EDC?style=flat&logo=pytest&logoColor=white)

Pipeline de Machine Learning para sinais financeiros que combina **fundamentos econômicos** de médio prazo (value, quality, growth, earnings surprise) com **microestrutura** de curto prazo (spread bid-ask, order imbalance, volume, volatilidade realizada), usando um protocolo de **validação cruzada purgada (Purged Group TimeSeriesSplit)** específico para séries financeiras — evitando o vazamento temporal que infla artificialmente a performance de modelos em backtests.

## 📦 Projeto

| Arquivo | O quê | Tecnologia |
|---|---|---|
| `src/cv.py` | Validação cruzada purgada (grupo + purge + embargo) | scikit-learn `BaseCrossValidator` |
| `src/data.py` | Geração do painel sintético (ativo × data) | pandas, numpy |
| `src/model.py` | Pipeline de treino/avaliação | scikit-learn |
| `src/main.py` | Roda tudo e compara com CV ingênua (com vazamento) | — |
| `tests/test_cv.py` | Testes unitários do splitter | pytest |

## 🧠 Conceitos implementados

- ✅ **Purge**: remove do treino qualquer observação cuja janela de rótulo (ex.: retorno nos próximos N pregões) se sobreponha à janela de teste.
- ✅ **Group-aware split**: cada grupo (uma data de pregão, com vários ativos naquele dia) fica inteiro de um lado só — nunca dividido entre treino e teste.
- ✅ **Embargo**: folga extra de pregões antes do teste, como margem de segurança contra autocorrelação residual entre observações vizinhas no tempo.
- ✅ **Expanding ou rolling window**: `max_train_group_size` opcional para limitar o treino a uma janela deslizante em vez de expandir indefinidamente.
- ✅ **Comparação com baseline ingênuo**: `src/main.py` roda o mesmo modelo com um `KFold` aleatório comum lado a lado, para quantificar o quanto o vazamento infla a métrica.

## 🚀 Como rodar localmente

### Pré-requisitos
- [Python 3.10+](https://www.python.org/downloads/)

### 1. Instalar dependências

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Rodar

```bash
# gera o painel sintético, roda a validação purgada e compara com a ingênua
python -m src.main
```

## 🔍 Testando

```bash
python -m pytest tests/ -q
```

6 testes cobrindo: nenhum grupo dividido entre treino/teste, treino sempre cronologicamente antes do teste, gap real de purge+embargo respeitado, limite de rolling window, erro ao chamar sem `groups`, e `get_n_splits`.

## 📊 Resultado de exemplo

```
Validação CORRETA — PurgedGroupTimeSeriesSplit
AUC médio (purged): ~0.50   (perto do acaso, como esperado em dados sintéticos)

Validação INGÊNUA — KFold aleatório (com vazamento de grupo/tempo)
AUC médio (naive): ~0.57   (inflada pelo vazamento)

Diferença (naive - purged) ≈ +0.07 -> vazamento infla a métrica
```

O ponto central do projeto não é o AUC em si — os dados são sintéticos — e sim mostrar **o quanto a escolha do protocolo de validação muda a métrica reportada**, um erro clássico (e caro) em backtests de modelos financeiros.

## ⚙️ Configuração

Os principais parâmetros ficam em `src/main.py` e `src/model.py`:

```python
label_horizon = 5     # quantos pregões à frente a label "enxerga"
n_splits = 5           # número de folds de teste
group_gap = 3            # folga de embargo, em número de pregões
```

## 🗺️ Estrutura

```
purged-cv-financeiro/
├── README.md
├── requirements.txt
├── src/
│   ├── cv.py            # PurgedGroupTimeSeriesSplit
│   ├── data.py            # gerador do painel sintético
│   ├── model.py             # pipeline de treino/avaliação
│   └── main.py                # pipeline fim-a-fim + comparação com CV ingênua
└── tests/
    └── test_cv.py              # testes unitários do splitter
```

## 🔮 Possíveis evoluções

- Trocar o painel sintético por dados reais (ex.: via `yfinance` ou uma base de fundamentos paga)
- Trocar o `GradientBoostingClassifier` por LightGBM/XGBoost
- Adicionar feature importance e um notebook de análise exploratória
- Empacotar o `PurgedGroupTimeSeriesSplit` como pacote instalável (`pip install -e .`)

---

© 2026 Gabriel Teramae Chan
