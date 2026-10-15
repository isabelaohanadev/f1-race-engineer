"""Corridas sintéticas de RESPOSTA CONHECIDA, com relógio coerente, paradas e perda no pit
plantadas.

O tempo de uma volta é `base + deslocamento[composto] + inclinação[composto] * idade - k * (volta -
1)
+ ruído`; a volta de entrada soma `in_extra` e a de saída (pneu novo e frio) soma `out_extra`, então
a
perda de uma parada plantada é `in_extra + out_extra` (mais termos de inclinação, que somem com
inclinação zero). O relógio da sessão é a soma dos tempos de volta mais a diferença de largada,
então
as diferenças entre carros (o que o undercut lê) são exatas, e as posições saem desse relógio.
"""

from dataclasses import dataclass, field

import numpy as np

from strategy.laps import LapRow

SLOPES = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}
OFFSETS = {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8}  # o composto mais duro é mais lento


@dataclass(frozen=True)
class Plan:
    """O que um piloto faz na corrida sintética."""

    code: str
    base: float = 90.0  # s: ritmo do piloto no composto de referência, idade 0
    compound: str = "MEDIUM"  # composto de largada
    age: int = 1  # idade do pneu na volta 1
    stops: tuple[tuple[int, str], ...] = ()  # (volta de ENTRADA nos boxes, novo composto)
    grid_gap: float = 0.0  # s atrás do primeiro na largada
    new_age: int = 1  # idade do pneu novo na volta de saída
    pace_noise: float | None = None  # sobrescreve o ruído da corrida só para este piloto


@dataclass(frozen=True)
class World:
    total_laps: int = 60
    slopes: dict[str, float] = field(default_factory=lambda: dict(SLOPES))
    offsets: dict[str, float] = field(default_factory=lambda: dict(OFFSETS))
    kappa: float = 0.05
    in_extra: float = 12.0
    out_extra: float = 9.0
    noise: float = 0.15
    seed: int = 0
    status_laps: dict[int, str] = field(default_factory=dict)  # volta -> status (para todos)
    # Sob Safety Car (status "4"): a volta de todos vai a `sc_factor` vezes o ritmo, e as sobretaxas
    # da entrada e da saída dos boxes valem `sc_pit_factor` vezes (a parada "barata").
    sc_factor: float = 1.0
    sc_pit_factor: float = 1.0


def build_race(plans: list[Plan], world: World | None = None) -> list[LapRow]:
    world = world or World()
    rng = np.random.default_rng(world.seed)
    per_driver: dict[str, list[tuple[int, float, str, int, int, bool, bool]]] = {}
    for plan in plans:
        stops = dict(plan.stops)
        compound, age, stint = plan.compound, plan.age, 1
        noise = world.noise if plan.pace_noise is None else plan.pace_noise
        laps = []
        next_compound: str | None = None  # não nulo na volta de SAÍDA dos boxes
        for n in range(1, world.total_laps + 1):
            pit_out = next_compound is not None
            if next_compound is not None:
                compound, age, stint = next_compound, plan.new_age, stint + 1
            pace = (
                plan.base
                + world.offsets[compound]
                + world.slopes[compound] * age
                - world.kappa * (n - 1)
                + rng.normal(0, noise)
            )
            pit_in = n in stops
            under_sc = "4" in world.status_laps.get(n, "1")
            pit_scale = world.sc_pit_factor if under_sc else 1.0
            seconds = pace * (world.sc_factor if under_sc else 1.0)
            seconds += pit_scale * (
                (world.in_extra if pit_in else 0) + (world.out_extra if pit_out else 0)
            )
            laps.append((n, seconds, compound, age, stint, pit_in, pit_out))
            next_compound = stops[n] if pit_in else None
            age += 1
        per_driver[plan.code] = laps

    rows: list[LapRow] = []
    clocks: dict[tuple[str, int], int] = {}
    for plan in plans:
        clock = round(plan.grid_gap * 1000)
        for n, seconds, compound, age, stint, pit_in, pit_out in per_driver[plan.code]:
            lap_ms = round(seconds * 1000)
            clock += lap_ms
            clocks[(plan.code, n)] = clock
            status = world.status_laps.get(n, "1")
            rows.append(
                LapRow(
                    plan.code, n, lap_ms, compound, age, stint, None, clock, pit_in, pit_out, status
                )
            )
    # posição na linha, pelo relógio, entre quem completou a volta
    ranked: dict[int, list[str]] = {}
    for lap in range(1, world.total_laps + 1):
        ranked[lap] = sorted(
            (code for (code, n) in clocks if n == lap), key=lambda c, lap=lap: clocks[(c, lap)]
        )
    return [
        LapRow(
            r.driver,
            r.lap_number,
            r.lap_time_ms,
            r.compound,
            r.tyre_life,
            r.stint,
            ranked[r.lap_number].index(r.driver) + 1,
            r.session_time_ms,
            r.pit_in,
            r.pit_out,
            r.track_status,
        )  # fmt: skip
        for r in rows
    ]


def chase_field(
    *,
    a_stop: int = 45,
    r_stop: int = 45,
    gap: float = 5.0,
    world: World | None = None,
    extra: tuple[Plan, ...] = (),
    slow_cars: int = 12,
) -> list[LapRow]:
    """O cenário do undercut: o rival `R` e o atacante `A`, igualmente rápidos (base 88 s) e ambos
    no primeiro stint de MEDIUM, com `A` a `gap` s atrás de `R`. Mais `slow_cars` carros lentos
    (base 92+), longe deles, que já pararam (MEDIUM ou SOFT para HARD nas voltas 12 a 23): dão ao
    modelo paradas para medir a perda e pares de compostos para medir a diferença entre eles.

    Os carros só interagem pelo relógio (não há tráfego no gerador), então a margem REALIZADA de um
    undercut é a diferença dos relógios no mundo contrafactual `chase_field(a_stop=N+1,
    r_stop=N+K)`, que tem a mesma história até a volta N."""
    world = world or World()
    plans = [
        Plan(
            f"D{i:02d}",
            base=92 + 0.3 * i,
            compound="MEDIUM" if i % 2 else "SOFT",
            stops=((12 + i, "HARD"),),
            grid_gap=3.0 * i,
        )
        for i in range(slow_cars)
    ]
    plans.append(Plan("R", base=88.0, stops=((r_stop, "HARD"),)))
    plans.append(Plan("A", base=88.0, stops=((a_stop, "HARD"),), grid_gap=gap))
    return build_race([*plans, *extra], world)


def clock_at(rows: list[LapRow], driver: str, lap: int) -> float:
    """O relógio da sessão (s) ao fim da volta, no gerador."""
    return (
        next(r.session_time_ms for r in rows if r.driver == driver and r.lap_number == lap) / 1000
    )
