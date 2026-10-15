"""Os parâmetros CONGELADOS do backtest. A explicação e as regras estão em
docs/backtest-protocolo.md.

POR QUE ESTE ARQUIVO. Um backtest só é uma avaliação limpa se as decisões do modelo foram fixadas
ANTES de ver o resultado. Estes valores foram fixados no commit que cria este arquivo, antes de
qualquer execução na população de corridas. `tests/strategy/test_protocol.py` prende cada um aos
padrões do código: mudar um padrão sem mudar aqui (e sem registrar no relatório como "ajuste
pós-resultado") quebra o teste, e mudar aqui depois de ver o backtest é um ajuste pós-resultado
e tem de aparecer com o antes e o depois.

O backtest NÃO lê os padrões dos módulos: monta `SimConfig` e `UndercutConfig` a partir destes
valores (`frozen_sim_config`, `frozen_undercut_config`).
"""

from dataclasses import dataclass

from strategy.montecarlo import SimConfig
from strategy.undercut import UndercutConfig

# O código em que os valores foram fixados (o último commit antes do congelamento).
FREEZE_BASE_COMMIT = "0169cde"


@dataclass(frozen=True, slots=True)
class Protocol:
    # ---- o alerta de undercut
    alert_threshold: float = 0.6  # liga
    alert_release: float = 0.5  # desliga (histerese)
    # fator de inflação do desvio da margem (provisório e congelado)
    undercut_sd_inflation: float = 1.3
    undercut_k_min: int = 3
    undercut_k_max: int = 10
    undercut_min_laps_left: int = 5
    # ---- as regras de parada, de ultrapassagem e do SC (heurísticas declaradas)
    pass_advantage_s: float = 0.7
    follow_gap_s: float = 0.7
    sc_wear: float = 0.3
    pit_decision_sd: float = 3.0
    max_stops: int = 3
    force_laps_left: int = 3
    min_stint_laps: int = 5
    max_sc: int = 3
    pseudo_events: float = 4.0
    start_jitter_laps: int = 3
    # ---- a simulação e a amostra
    n_sims: int = 5000
    first_lap: int = 5  # voltas avaliadas: first_lap, first_lap + every, ... até a penúltima
    every: int = 5
    sc_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    # ---- o erro do tempo de volta
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5)  # voltas à frente
    interval_coverage: float = 0.90
    # ---- o estado no instante t
    max_silence_laps: float = 2.0  # sem cruzar a linha por tantas voltas típicas = fora da corrida
    # ---- a estatística
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0
    confidence: float = 0.95
    headline_variant: str = "encolhimento"  # o `k` do resultado principal
    variants: tuple[str, ...] = ("fisico", "estimado", "encolhimento")


FROZEN = Protocol()


def frozen_undercut_config(protocol: Protocol = FROZEN) -> UndercutConfig:
    return UndercutConfig(
        threshold=protocol.alert_threshold,
        release=protocol.alert_release,
        k_min=protocol.undercut_k_min,
        k_max=protocol.undercut_k_max,
        min_laps_left=protocol.undercut_min_laps_left,
        pass_advantage_s=protocol.pass_advantage_s,
        follow_gap_s=protocol.follow_gap_s,
        sd_inflation=protocol.undercut_sd_inflation,
    )


def frozen_sim_config(protocol: Protocol = FROZEN, **overrides) -> SimConfig:
    """O `SimConfig` do protocolo. `overrides` só para o que o protocolo varia de propósito (o
    multiplicador da taxa de SC da sensibilidade, o número de simulações num teste)."""
    return SimConfig(
        n_sims=overrides.pop("n_sims", protocol.n_sims),
        sc_wear=protocol.sc_wear,
        pit_decision_sd=protocol.pit_decision_sd,
        max_stops=protocol.max_stops,
        force_laps_left=protocol.force_laps_left,
        min_stint_laps=protocol.min_stint_laps,
        max_sc=protocol.max_sc,
        pseudo_events=protocol.pseudo_events,
        start_jitter_laps=protocol.start_jitter_laps,
        traffic=frozen_undercut_config(protocol),
        **overrides,
    )
