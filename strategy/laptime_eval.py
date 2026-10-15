"""O erro de previsão do TEMPO DE VOLTA: o modelo acerta o ritmo que os pilotos fazem a seguir?

É a pergunta mais baixa da cadeia (antes do undercut e do Monte Carlo) e a mais fácil de medir: no
instante em que o líder completa a volta N, o modelo prevê as próximas `h` voltas de cada piloto
(h = 1 a 5) e a previsão é comparada com a volta que ele de fato fez.

O QUE ENTRA. Só voltas que o modelo se propõe a prever: limpas (pista verde, sem entrada nem saída
dos
boxes, depois da volta 1), de seco, no MESMO stint em que o piloto estava em t. A parada, o safety
car e
a mudança de composto são outras perguntas (as do undercut e do Monte Carlo).

O QUE SE MEDE, por horizonte `h` (voltas à frente da última volta completa de cada piloto):
* viés e erro absoluto médio (s), previsto menos observado;
* a COBERTURA do intervalo de 90%: com o desvio da previsão (`sqrt(sd_nivel_e_inclinacao^2 +
  sigma_volta^2)`, a incerteza do modelo, não a do ruído sozinho), a fração das voltas observadas
  que cai dentro de 1,645 desvios. O nominal é 90%;
* o COMPARADOR DE PERSISTÊNCIA: a mediana das 3 últimas voltas limpas do próprio piloto, sem modelo
  nenhum. O modelo só vale a pena se errar menos que isto.

UNIDADE. A corrida: a média por corrida (das voltas e dos pilotos) é o que o bootstrap ressorteia,
pelo
mesmo motivo do Brier (as voltas de uma corrida são correlacionadas).

ESTADO. `snapshot="tempo"` (o padrão): o instante em que o líder completa a volta N
(`strategy.snapshot`); `"volta"`, o desenho antigo, só para medir o vazamento.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist, median

import numpy as np

from strategy.fuel import FuelVariant, laps_until
from strategy.laps import DRY_COMPOUNDS, LapRow
from strategy.pace import DEFAULT_SIGMA, fit_pace
from strategy.priors import DEFAULT_PRIORS, Priors
from strategy.snapshot import MAX_SILENCE_LAPS, car_states, state_at_leader_lap

BASELINE_LAPS = 3  # as últimas voltas limpas do comparador de persistência


@dataclass(frozen=True, slots=True)
class LapTimeRecord:
    """O resumo de UMA (corrida, volta N, horizonte): a média sobre os pilotos que entram."""

    race: str
    lap: int
    horizon: int
    n: int  # pilotos com a volta prevista e a observada
    bias: float  # previsto menos observado, s
    mae: float
    coverage: float  # fração das observadas dentro do intervalo
    baseline_bias: (
        float | None
    )  # a persistência; None se nenhum piloto tinha 3 voltas de referência
    baseline_mae: float | None
    baseline_n: int


def is_clean(row: LapRow) -> bool:
    """A volta que o modelo se propõe a prever."""
    return (
        row.lap_time_ms is not None
        and row.track_status == "1"
        and not row.pit_in
        and not row.pit_out
        and row.lap_number > 1
        and row.compound in DRY_COMPOUNDS
    )


def persistence(rows: Sequence[LapRow], driver: str, upto: int) -> float | None:
    """A mediana das últimas voltas limpas do piloto até a volta `upto` (inclusive), sem modelo."""
    clean = sorted(
        (r for r in rows if r.driver == driver and r.lap_number <= upto and is_clean(r)),
        key=lambda r: r.lap_number,
    )[-BASELINE_LAPS:]
    return float(median(r.lap_time_ms / 1000 for r in clean)) if clean else None  # type: ignore[operator]


def lap_time_records(
    rows: Sequence[LapRow],
    *,
    race: str,
    total_laps: int,
    laps: Sequence[int],
    horizons: Sequence[int],
    priors: Priors = DEFAULT_PRIORS,
    variant: FuelVariant = FuelVariant.ENCOLHIMENTO,
    snapshot: str = "tempo",
    coverage: float = 0.90,
    max_silence_laps: float = MAX_SILENCE_LAPS,
) -> list[LapTimeRecord]:
    """Os registros de cada volta N de `laps` e cada horizonte de `horizons`."""
    z = NormalDist().inv_cdf((1 + coverage) / 2)
    actual = {(r.driver, r.lap_number): r for r in rows}
    records: list[LapTimeRecord] = []
    for lap in laps:
        if snapshot == "tempo":
            state = state_at_leader_lap(rows, lap, max_silence_laps=max_silence_laps)
            if state is None:
                continue
            visible, cars = list(state.rows), state.cars
        else:
            visible = laps_until(rows, lap)
            cars = car_states(visible, lap)
        pace = fit_pace(
            visible, upto_lap=lap, total_laps=total_laps, variant=variant, priors=priors
        )
        errors: dict[int, list[tuple[float, float, bool]]] = {h: [] for h in horizons}
        naive: dict[int, list[float]] = {h: [] for h in horizons}
        for driver, car in cars.items():
            level = pace.current(driver)
            if level is None or level.stint != car.stint or car.age is None:
                continue
            sigma = level.slope.sigma_lap or pace.degradation.sigma_lap or DEFAULT_SIGMA
            base = persistence(visible, driver, car.laps_done)
            for h in horizons:
                target = actual.get((driver, car.laps_done + h))
                if target is None or not is_clean(target) or target.stint != car.stint:
                    continue
                age = car.age + h
                predicted = level.predict(age, car.laps_done + h, pace.kappa)
                sd = float(np.hypot(level.sd_of_sum([age]), sigma))
                observed = target.lap_time_ms / 1000  # type: ignore[operator]
                error = predicted - observed
                errors[h].append((error, abs(error), abs(error) <= z * sd))
                if base is not None:
                    naive[h].append(base - observed)
        for h in horizons:
            if not errors[h]:
                continue
            arr = np.array(errors[h], dtype=float)
            naive_errors = np.array(naive[h], dtype=float)
            records.append(
                LapTimeRecord(
                    race=race,
                    lap=lap,
                    horizon=h,
                    n=len(arr),
                    bias=float(arr[:, 0].mean()),
                    mae=float(arr[:, 1].mean()),
                    coverage=float(arr[:, 2].mean()),
                    baseline_bias=float(naive_errors.mean()) if naive_errors.size else None,
                    baseline_mae=float(np.abs(naive_errors).mean()) if naive_errors.size else None,
                    baseline_n=int(naive_errors.size),
                )
            )
    return records


def per_race(records: Sequence[LapTimeRecord], horizon: int, attribute: str) -> dict[str, float]:
    """A média de `attribute` por corrida, nas voltas avaliadas, ponderando cada volta pelo seu `n`
    (a corrida é a unidade que o bootstrap ressorteia; dentro dela, um piloto-volta pesa igual)."""
    sums: dict[str, list[tuple[float, int]]] = {}
    for r in records:
        value = getattr(r, attribute)
        if r.horizon == horizon and value is not None:
            weight = r.baseline_n if attribute.startswith("baseline") else r.n
            sums.setdefault(r.race, []).append((float(value), weight))
    return {
        race: sum(v * w for v, w in vs) / sum(w for _, w in vs)
        for race, vs in sums.items()
        if sum(w for _, w in vs) > 0
    }
