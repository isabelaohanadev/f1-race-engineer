"""Testes das constantes sem a corrida avaliada (strategy/priors.py): leave-one-race-out."""

from dataclasses import replace

import pytest

from strategy.priors import (
    DEFAULT_PRIORS,
    MIN_RACES,
    PIT_LOSS_BETWEEN_FLOOR,
    RaceStats,
    circuit_history,
    leave_one_out,
)


def stat(label, year=2024, round_=1, circuit="Sakhir", **kw) -> RaceStats:
    fields = {
        "tau": 0.02,
        "between_compounds": 0.015,
        "pit_loss": 22.0,
        "pit_loss_sd": 1.4,
        "n_green_stops": 20,
    }
    return RaceStats(label, circuit, year, round_, **(fields | kw))


def season() -> list[RaceStats]:
    """Cinco corridas com perdas conhecidas (20, 22, 24, 26, 28) em circuitos diferentes."""
    return [
        stat(f"R{i}", round_=i, circuit=f"C{i}", pit_loss=20.0 + 2 * (i - 1), tau=0.01 * i)
        for i in range(1, 6)
    ]


# ------------------------------------------------------------------ deixar a corrida de fora


def test_the_excluded_race_has_no_influence_on_its_own_constants():
    """O ponto do leave-one-race-out: o que a corrida avaliada diz de si NÃO entra na constante
    que será usada para avaliá-la. Trocar os números dela por absurdos não muda nada."""
    stats = season()
    honest = leave_one_out(stats, "R3")
    stats[2] = replace(stats[2], pit_loss=999.0, tau=9.9, between_compounds=9.9, pit_loss_sd=99.0)

    assert leave_one_out(stats, "R3") == honest


def test_the_other_races_do_move_the_constants():
    stats = season()
    base = leave_one_out(stats, "R3")
    stats[0] = replace(stats[0], pit_loss=100.0)

    assert leave_one_out(stats, "R3").pit_loss.mean != base.pit_loss.mean


def test_the_constants_are_the_medians_of_the_other_races():
    priors = leave_one_out(season(), "R3")  # sobram 20, 22, 26, 28

    assert priors.pit_loss.mean == pytest.approx(24.0)  # mediana de 20, 22, 26, 28
    assert priors.tau == pytest.approx(0.03)  # mediana de 0,01, 0,02, 0,04, 0,05
    assert priors.pit_loss.stop_sd == pytest.approx(1.4)


def test_the_between_race_spread_is_robust_and_never_below_the_floor():
    flat = [stat(f"R{i}", round_=i, circuit=f"C{i}", pit_loss=22.0) for i in range(1, 7)]

    assert leave_one_out(flat, "R1").pit_loss.between_sd == PIT_LOSS_BETWEEN_FLOOR
    spread = leave_one_out(season(), "R3").pit_loss.between_sd
    assert spread > PIT_LOSS_BETWEEN_FLOOR


def test_races_with_unknown_values_are_ignored_not_counted_as_zero():
    stats = season()
    stats[3] = replace(stats[3], tau=None, pit_loss=None, pit_loss_sd=None)  # R4 sem dados

    priors = leave_one_out(stats, "R3")  # sobram R1, R2, R5 com valores: perdas 20, 22, 28

    assert priors.pit_loss.mean == pytest.approx(22.0)  # contada como zero daria 21
    assert priors.tau == pytest.approx(0.02)  # taus 0,01, 0,02, 0,05; com um zero seria 0,015


def test_with_too_few_reference_races_the_default_applies():
    two = [stat("A", round_=1, circuit="a"), stat("B", round_=2, circuit="b")]

    priors = leave_one_out(two, "A")

    assert MIN_RACES == 3
    assert priors.tau == DEFAULT_PRIORS.tau and priors.pit_loss == DEFAULT_PRIORS.pit_loss


def test_the_source_says_which_races_were_used():
    priors = leave_one_out(season(), "R3")

    assert "sem R3" in priors.source and "4 corridas" in priors.source
    assert "INCLUSIVE a avaliada" in DEFAULT_PRIORS.source  # o padrão avisa que não é para backtest


# ------------------------------------------------------------------ só as anteriores


def test_earlier_only_keeps_just_the_races_before_the_target():
    stats = season()

    priors = leave_one_out(stats, "R4", earlier_only=True)  # só R1, R2, R3: 20, 22, 24

    assert priors.pit_loss.mean == pytest.approx(22.0)
    assert "só anteriores" in priors.source


def test_earlier_only_ignores_later_races_completely():
    stats = season()
    honest = leave_one_out(stats, "R3", earlier_only=True)
    stats[4] = replace(stats[4], pit_loss=500.0)  # a corrida POSTERIOR muda

    assert leave_one_out(stats, "R3", earlier_only=True) == honest


def test_earlier_only_for_an_unknown_race_is_an_error():
    with pytest.raises(ValueError, match="earlier_only"):
        leave_one_out(season(), "R99", earlier_only=True)


def test_orders_by_year_and_round_not_by_list_position():
    stats = [
        stat("new", year=2025, round_=1, circuit="a"),
        stat("old", year=2024, round_=24, circuit="b"),
    ]

    assert [s.label for s in circuit_history(stats, "new")] == []  # circuitos diferentes
    same = [stat("new", year=2025, round_=1), stat("old", year=2024, round_=24)]
    assert [s.label for s in circuit_history(same, "new")] == ["old"]


# ------------------------------------------------------------------ o circuito


def test_circuit_history_has_only_earlier_editions_of_the_same_circuit():
    stats = [
        stat("2023", year=2023, round_=1, pit_loss=24.0),
        stat("2024", year=2024, round_=1, pit_loss=25.0),
        stat("2025", year=2025, round_=1, pit_loss=26.0),  # a corrida avaliada
        stat("2026", year=2026, round_=1, pit_loss=99.0),  # POSTERIOR: nunca
        stat("outra", year=2024, round_=5, circuit="Jeddah", pit_loss=18.0),
    ]

    history = circuit_history(stats, "2025")

    assert [s.label for s in history] == ["2023", "2024"]


def test_a_later_edition_of_the_circuit_never_enters_the_prior():
    stats = [
        stat("2023", year=2023, pit_loss=24.0),
        stat("2024", year=2024, pit_loss=26.0),
        stat("outra1", year=2024, round_=2, circuit="a", pit_loss=20.0),
        stat("outra2", year=2024, round_=3, circuit="b", pit_loss=21.0),
        stat("outra3", year=2024, round_=4, circuit="c", pit_loss=22.0),
    ]
    honest = leave_one_out(stats, "2024")
    stats.append(stat("2025", year=2025, pit_loss=1.0))  # edição POSTERIOR do mesmo circuito

    assert leave_one_out(stats, "2024").pit_loss.mean == honest.pit_loss.mean == 24.0
    assert "centro do circuito (1 edições anteriores)" in honest.source


def test_the_circuit_history_centres_the_prior_instead_of_the_other_circuits():
    stats = [stat("antes", year=2023, pit_loss=29.0), stat("agora", year=2024, pit_loss=29.5)]
    stats += [
        stat(f"x{i}", year=2024, round_=i + 2, circuit=f"c{i}", pit_loss=21.0) for i in range(4)
    ]

    assert leave_one_out(stats, "agora").pit_loss.mean == pytest.approx(
        29.0
    )  # a pista, não a média 21


def test_a_race_without_a_circuit_has_no_history():
    assert circuit_history([stat("a", circuit=None), stat("b", year=2023, circuit=None)], "a") == []
    assert circuit_history([stat("a")], "inexistente") == []


def test_a_previous_edition_without_a_measured_loss_is_not_history():
    stats = [stat("2023", year=2023, pit_loss=None), stat("2024", year=2024)]

    assert circuit_history(stats, "2024") == []
