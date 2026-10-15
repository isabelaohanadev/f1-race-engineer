"""O protocolo congelado do backtest (strategy/protocol.py) bate com os padrões do código.

Se um destes testes quebra, ou um padrão mudou sem querer (e o backtest não mudaria, porque lê o
protocolo), ou alguém mudou o protocolo: isso é um AJUSTE PÓS-RESULTADO e tem de ser registrado no
relatório do backtest com o antes e o depois.
"""

import dataclasses

from strategy.montecarlo import SimConfig
from strategy.protocol import (
    FROZEN,
    Protocol,
    frozen_sim_config,
    frozen_undercut_config,
)
from strategy.undercut import UndercutConfig


def test_the_frozen_undercut_config_is_the_code_default():
    assert frozen_undercut_config() == UndercutConfig()


def test_the_frozen_simulation_config_is_the_code_default():
    assert frozen_sim_config() == SimConfig()


def test_the_values_the_documents_quote_are_the_frozen_ones():
    """Os números que docs/backtest-protocolo.md cita."""
    assert FROZEN.alert_threshold == 0.6 and FROZEN.alert_release == 0.5
    assert FROZEN.undercut_sd_inflation == 1.3
    assert (FROZEN.pass_advantage_s, FROZEN.follow_gap_s) == (0.7, 0.7)
    assert (FROZEN.sc_wear, FROZEN.pit_decision_sd) == (0.3, 3.0)
    assert (FROZEN.max_stops, FROZEN.force_laps_left, FROZEN.min_stint_laps) == (3, 3, 5)
    assert FROZEN.n_sims == 5000 and FROZEN.sc_multipliers == (0.5, 1.0, 2.0)
    assert (FROZEN.first_lap, FROZEN.every) == (5, 5)
    assert FROZEN.variants == ("fisico", "estimado", "encolhimento")
    assert FROZEN.headline_variant == "encolhimento"


def test_the_protocol_is_immutable():
    assert dataclasses.is_dataclass(FROZEN)
    try:
        FROZEN.alert_threshold = 0.7  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("o protocolo congelado não pode ser alterado em tempo de execução")


def test_the_overrides_are_only_for_what_the_protocol_varies_on_purpose():
    config = frozen_sim_config(n_sims=100, sc_rate_multiplier=2.0)

    assert config.n_sims == 100 and config.sc_rate_multiplier == 2.0
    assert config.sc_wear == FROZEN.sc_wear and config.traffic.pass_advantage_s == 0.7


def test_a_modified_protocol_builds_a_different_config():
    """O backtest lê o protocolo, não os padrões: um protocolo alterado muda o que ele roda."""
    changed = Protocol(alert_threshold=0.7, undercut_sd_inflation=1.5)

    assert frozen_undercut_config(changed).threshold == 0.7
    assert frozen_undercut_config(changed).sd_inflation == 1.5
    assert frozen_sim_config(changed).traffic.threshold == 0.7
