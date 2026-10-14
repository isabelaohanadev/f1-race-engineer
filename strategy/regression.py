"""Mínimos quadrados em numpy, com o que o modelo de pneu precisa e um OLS "de livro" não dá.

Por que não statsmodels/scipy (o pedido foi "só se justificar"): são ~60 linhas de álgebra linear, e
escrevê-las deixa explícito cada ponto em que a estimativa pode enganar. statsmodels traria um
pacote grande para duas coisas que aqui cabem num arquivo: (1) um ajuste ROBUSTO a voltas ruins e
(2) um erro-padrão HONESTO para dados que não são independentes.

* ROBUSTO (Huber, reponderação iterativa). Mesmo com a limpeza, sobram voltas ruins: tráfego, um
  erro do piloto, um bloqueio de roda. Mínimos quadrados comuns elevam o erro ao quadrado, então UMA
  volta 5 s mais lenta puxa a reta inteira. Huber trata o erro como quadrático até `k` desvios
  robustos e como linear depois: o que é ruído manda no ajuste, o que é erro grosseiro perde peso.
* ERRO-PADRÃO POR CLUSTER (piloto). As voltas de um piloto não são independentes: o tráfego e o
  ritmo do dia se repetem de uma volta para a seguinte. A fórmula clássica trata as ~1.000 voltas
  como 1.000 observações independentes e devolve um erro-padrão pequeno demais, e é esse número que
  decide se uma estimativa é confiável. O estimador "sanduíche" por cluster agrupa os resíduos por
  piloto (são ~20 observações de fato independentes) e corrige o tamanho da amostra.
* ESTIMABILIDADE. Se uma coluna é combinação das outras (idade do pneu e volta da corrida quando
  todos os stints começam na mesma volta), o coeficiente dela NÃO EXISTE, e o `lstsq` devolve um
  número qualquer sem reclamar. `identified` diz, coluna a coluna, se o coeficiente é determinado
  pelos dados.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Constante de Huber clássica: ~95% de eficiência se o ruído for normal.
HUBER_K = 1.345
# MAD * 1,4826 = desvio-padrão, se a distribuição for normal.
_MAD_TO_SIGMA = 1.4826


# t de Student bicaudal a 95%, por graus de liberdade (1..30). Uma tabela de 30 números em vez de
# trazer o scipy para um único quantil; acima de 30 a diferença para a normal é menor que 4%.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def t_critical_95(dof: int) -> float:
    if dof < 1:
        return float("nan")
    return _T975[dof - 1] if dof <= len(_T975) else 1.96


@dataclass(frozen=True, slots=True)
class LinearFit:
    coef: np.ndarray
    se: np.ndarray  # erro-padrão por cluster; NaN onde o coeficiente não é estimável
    identified: np.ndarray  # por coluna: o dado determina este coeficiente?
    n_obs: int
    n_clusters: int
    rank: int
    weights: np.ndarray  # peso de Huber de cada observação (1 = confiada, <1 = outlier)

    def ci95(self, index: int) -> tuple[float, float]:
        """Intervalo de 95%. Com ~20 clusters a normal (1,96) é otimista: usa a t de Student com
        n_clusters - 1 graus de liberdade."""
        half = t_critical_95(self.n_clusters - 1) * self.se[index]
        return float(self.coef[index] - half), float(self.coef[index] + half)


def _estimable_columns(X: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """A coluna j é estimável se e_j está no espaço-linha de X, o que equivale a P[j, j] = 1 para
    P = pinv(X) @ X (o projetor ortogonal sobre esse espaço)."""
    projector = np.linalg.pinv(X) @ X
    return np.diag(projector) > 1 - tol


def fit_linear(
    X: np.ndarray,
    y: np.ndarray,
    *,
    clusters: Sequence | None = None,
    huber_k: float | None = HUBER_K,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> LinearFit:
    """Ajusta y = X b. `clusters`: um rótulo por observação (os erros se correlacionam DENTRO dele).
    `huber_k=None` dá mínimos quadrados comuns."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X {X.shape} e y {y.shape} não casam")
    n, k = X.shape
    if clusters is not None and len(clusters) != n:
        raise ValueError("clusters precisa ter um rótulo por observação")

    weights = np.ones(n)
    quadratic = np.ones(n)  # ψ'(u): 1 na parte quadrática da perda de Huber, 0 na linear
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    if huber_k is not None:
        for _ in range(max_iter):
            resid = y - X @ coef
            scale = _MAD_TO_SIGMA * np.median(np.abs(resid - np.median(resid)))
            if scale < 1e-12:  # ajuste (quase) perfeito: não há o que reponderar
                break
            u = np.abs(resid) / scale
            quadratic = (u <= huber_k).astype(float)
            weights = np.where(u <= huber_k, 1.0, huber_k / np.maximum(u, 1e-12))
            sw = np.sqrt(weights)
            updated = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
            moved = np.max(np.abs(updated - coef)) >= tol * (1 + np.max(np.abs(coef)))
            coef = updated
            if not moved:
                break

    resid = y - X @ coef
    rank = int(np.linalg.matrix_rank(X))
    identified = _estimable_columns(X)

    if clusters is None:
        codes = np.arange(n)
    else:
        _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(codes.max()) + 1 if n else 0

    se = np.full(k, np.nan)
    if n_clusters > 1 and n > rank:
        # O pão do estimador-M usa a DERIVADA ψ' da perda (1 ou 0), não os pesos da reponderação:
        # os pesos dão voz a uma volta ruim que a perda de Huber na verdade só "empurra" com força
        # constante, e o erro-padrão sairia pequeno demais (medido: 88% de cobertura em vez de 95%).
        bread = np.linalg.pinv(X.T @ (X * quadratic[:, None]))
        scores = np.zeros((n_clusters, k))
        np.add.at(scores, codes, X * (weights * resid)[:, None])
        # Correção de amostra pequena (a mesma do "CR1"): poucos clusters e muitos parâmetros fariam
        # a variância sair pequena demais.
        adjust = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - rank))
        variance = adjust * bread @ (scores.T @ scores) @ bread
        se = np.sqrt(np.maximum(np.diag(variance), 0.0))
    se = np.where(identified, se, np.nan)

    return LinearFit(coef, se, identified, n, n_clusters, rank, weights)


def one_hot(labels: Sequence, *, drop_first: bool = False) -> tuple[np.ndarray, list]:
    """Colunas 0/1, uma por rótulo (em ordem), e a lista dos rótulos. `drop_first` omite o primeiro
    (a categoria de referência), necessário quando o modelo já tem um intercepto implícito."""
    names = sorted(set(labels))
    kept = names[1:] if drop_first else names
    matrix = np.array([[1.0 if label == name else 0.0 for name in kept] for label in labels])
    return matrix.reshape(len(labels), len(kept)), kept
