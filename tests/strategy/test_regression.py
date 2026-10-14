"""Testes da regressão (strategy/regression.py) com dados SINTÉTICOS de coeficiente conhecido.

A resposta certa é plantada nos dados: se o estimador não a recupera, o defeito é dele.
"""

import numpy as np
import pytest

from strategy.regression import HUBER_K, fit_linear, one_hot, t_critical_95


def design(rng: np.random.Generator, n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = rng.uniform(0, 30, n)
    x2 = rng.uniform(0, 60, n)
    X = np.column_stack([np.ones(n), x1, x2])
    return X, x1, x2


def test_recovers_the_exact_coefficients_when_there_is_no_noise():
    rng = np.random.default_rng(1)
    X, x1, x2 = design(rng)
    y = 2.0 + 0.5 * x1 - 3.0 * x2

    fit = fit_linear(X, y)

    np.testing.assert_allclose(fit.coef, [2.0, 0.5, -3.0], atol=1e-8)
    assert fit.identified.all() and fit.rank == 3


def test_recovers_a_known_slope_within_its_own_error_bar():
    rng = np.random.default_rng(2)
    X, x1, x2 = design(rng, 800)
    y = 90.0 + 0.07 * x1 - 0.05 * x2 + rng.normal(0, 0.4, len(x1))

    fit = fit_linear(X, y, clusters=rng.integers(0, 20, len(y)))

    assert abs(fit.coef[1] - 0.07) < 3 * fit.se[1]
    assert abs(fit.coef[2] + 0.05) < 3 * fit.se[2]
    assert 0 < fit.se[1] < 0.01  # e o erro-padrão é pequeno, não um número qualquer


@pytest.mark.parametrize("huber_k", [None, HUBER_K], ids=["ols", "huber"])
def test_the_95_percent_interval_covers_the_truth_about_95_percent_of_the_time(huber_k):
    """Teste de calibração do erro-padrão: ele só serve se o intervalo cobre a verdade na proporção
    que promete. 400 simulações com 20 clusters (o número de pilotos de uma corrida) e um erro
    comum a cada cluster. Foi este teste que pegou o pão errado do sanduíche de Huber (cobertura de
    88%) e a normal otimista demais para 20 clusters."""
    rng = np.random.default_rng(3)
    hits = 0
    runs = 400
    for _ in range(runs):
        X, x1, x2 = design(rng, 400)
        clusters = rng.integers(0, 20, len(x1))
        shock = rng.normal(0, 0.3, 20)[clusters]  # erro comum a todo o cluster (o ritmo do piloto)
        y = 90.0 + 0.07 * x1 - 0.05 * x2 + shock + rng.normal(0, 0.4, len(x1))
        low, high = fit_linear(X, y, clusters=clusters, huber_k=huber_k).ci95(1)
        hits += low <= 0.07 <= high

    assert 0.915 <= hits / runs <= 0.975


def test_t_critical_value_matches_the_textbook_and_falls_back_to_the_normal():
    assert t_critical_95(19) == pytest.approx(2.093)
    assert t_critical_95(1) == pytest.approx(12.706)
    assert t_critical_95(30) == pytest.approx(2.042)
    assert t_critical_95(500) == 1.96
    assert np.isnan(t_critical_95(0))


def test_cluster_robust_error_is_larger_than_the_naive_one_when_errors_are_correlated():
    rng = np.random.default_rng(4)
    n_clusters, per = 20, 40
    clusters = np.repeat(np.arange(n_clusters), per)
    # O regressor varia ENTRE clusters (como a volta em que cada piloto começa o stint) e o erro é
    # comum ao cluster: tratar as 800 voltas como independentes subestima a incerteza.
    x = np.repeat(rng.uniform(0, 30, n_clusters), per)
    y = 0.07 * x + np.repeat(rng.normal(0, 1.0, n_clusters), per) + rng.normal(0, 0.1, len(x))
    X = np.column_stack([np.ones(len(x)), x])

    naive = fit_linear(X, y, huber_k=None)  # cada observação é o seu próprio cluster
    clustered = fit_linear(X, y, clusters=clusters, huber_k=None)

    assert clustered.se[1] > 2 * naive.se[1]


def test_huber_is_far_less_moved_by_gross_outliers_than_ordinary_least_squares():
    """Voltas ruins que se concentram no fim do stint (tráfego com o pneu gasto) enviesam o OLS:
    contaminação ALEATÓRIA não enviesaria a inclinação, então ela é plantada onde a idade é alta."""
    rng = np.random.default_rng(5)
    X, x1, x2 = design(rng, 600)
    y = 90.0 + 0.07 * x1 - 0.05 * x2 + rng.normal(0, 0.3, len(x1))
    late = np.flatnonzero(x1 > 22)
    bad = rng.choice(late, 30, replace=False)  # ~5% das voltas, 6 s mais lentas
    y[bad] += 6.0

    ols = fit_linear(X, y, huber_k=None)
    huber = fit_linear(X, y, huber_k=HUBER_K)

    assert abs(ols.coef[1] - 0.07) > 0.01  # premissa: o OLS ficou mesmo enviesado
    assert abs(huber.coef[1] - 0.07) < abs(ols.coef[1] - 0.07) / 4
    assert huber.weights[bad].mean() < 0.2  # as voltas ruins perderam quase todo o peso
    assert huber.weights.mean() > 0.9


def test_columns_in_a_linear_dependency_are_reported_as_not_identified():
    """O caso do combustível: se todo stint começa na mesma volta, `lap = age + 1`. A dependência
    (lap - age - 1 = 0) envolve as TRÊS colunas, intercepto incluído: nenhuma é estimável."""
    rng = np.random.default_rng(6)
    age = rng.uniform(1, 30, 300)
    X = np.column_stack([np.ones(300), age, age + 1])
    y = 90 + 0.05 * age + rng.normal(0, 0.2, 300)

    fit = fit_linear(X, y, clusters=rng.integers(0, 10, 300))

    assert not fit.identified.any()
    assert np.isnan(fit.se).all()
    assert fit.rank == 2


def test_a_column_outside_the_dependency_stays_identified():
    rng = np.random.default_rng(10)
    age = rng.uniform(1, 30, 300)
    other = rng.uniform(0, 5, 300)
    X = np.column_stack([np.ones(300), age, 2 * age, other])  # 2*age duplica age; `other` é livre

    fit = fit_linear(
        X, 1 + 0.5 * other + rng.normal(0, 0.1, 300), clusters=rng.integers(0, 10, 300)
    )

    assert fit.identified.tolist() == [True, False, False, True]
    assert abs(fit.coef[3] - 0.5) < 3 * fit.se[3]


def test_it_becomes_identified_as_soon_as_the_columns_stop_moving_together():
    rng = np.random.default_rng(7)
    age = rng.uniform(1, 30, 300)
    lap = age + rng.integers(0, 40, 300)  # cada stint começa numa volta diferente
    X = np.column_stack([np.ones(300), age, lap])
    y = 90 + 0.08 * age - 0.05 * lap + rng.normal(0, 0.2, 300)

    fit = fit_linear(X, y, clusters=rng.integers(0, 10, 300))

    assert fit.identified.all()
    assert abs(fit.coef[1] - 0.08) < 3 * fit.se[1] and abs(fit.coef[2] + 0.05) < 3 * fit.se[2]


def test_the_error_is_undefined_with_a_single_cluster_or_without_spare_observations():
    rng = np.random.default_rng(8)
    X, x1, _ = design(rng, 50)
    y = 1 + x1

    one_cluster = fit_linear(X, y, clusters=np.zeros(50, dtype=int))
    saturated = fit_linear(X[:3], y[:3])  # 3 observações, 3 parâmetros: nada sobra para o erro

    assert np.isnan(one_cluster.se).all()
    assert np.isnan(saturated.se).all()


def test_it_is_deterministic():
    rng = np.random.default_rng(9)
    X, x1, x2 = design(rng)
    y = 1 + 0.1 * x1 + rng.normal(0, 1, len(x1))
    clusters = rng.integers(0, 12, len(y))

    a, b = fit_linear(X, y, clusters=clusters), fit_linear(X, y, clusters=clusters)

    np.testing.assert_array_equal(a.coef, b.coef)
    np.testing.assert_array_equal(a.se, b.se)


@pytest.mark.parametrize(
    ("X", "y", "clusters"),
    [
        (np.ones((5, 2)), np.ones(4), None),
        (np.ones(5), np.ones(5), None),
        (np.ones((5, 2)), np.ones(5), [0, 1]),
    ],
)
def test_inconsistent_shapes_are_rejected_loudly(X, y, clusters):
    with pytest.raises(ValueError):
        fit_linear(X, y, clusters=clusters)


# ------------------------------------------------------------------ one_hot


def test_one_hot_builds_one_column_per_label_in_sorted_order():
    matrix, names = one_hot(["VER", "NOR", "VER", "HAM"])

    assert names == ["HAM", "NOR", "VER"]
    assert matrix.tolist() == [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]]


def test_one_hot_drop_first_removes_the_reference_category():
    matrix, names = one_hot(["SOFT", "HARD", "MEDIUM"], drop_first=True)

    assert names == ["MEDIUM", "SOFT"]  # HARD (o primeiro, em ordem) é a referência
    assert matrix.tolist() == [[0, 1], [0, 0], [1, 0]]


def test_one_hot_with_a_single_label_and_drop_first_has_no_columns():
    matrix, names = one_hot(["SOFT", "SOFT"], drop_first=True)

    assert matrix.shape == (2, 0) and names == []
