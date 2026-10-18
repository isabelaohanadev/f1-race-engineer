"""O resultado de estratégia como ele viaja: um dicionário JSON versionado (`v`).

É o MESMO documento no hash do Redis (que o `GET /races/{y}/{r}/strategy` lê), no stream de
resultados (que o hub repassa ao WebSocket como evento `strategy`) e na resposta da API. Um formato
só: o que a API serve é, byte a byte, o que o consumidor gravou.

VERSÃO (`v`). Campo novo e opcional não muda a versão (quem lê ignora o que não conhece); mudar o
significado de um campo, ou removê-lo, sim. O cliente que vê um `v` maior que o que conhece deve
mostrar "versão não suportada" em vez de adivinhar.

HONESTIDADE DO PAYLOAD (a pedido, depois do backtest: `docs/backtest.md`). O Monte Carlo NÃO supera
"o líder atual vence" em nenhuma fase da corrida e é pior nos últimos 20%. Por isso cada previsão
traz, ao lado da probabilidade do Monte Carlo, a do comparador (`baseline_win`: 1 para quem lidera
agora, 0 para os outros), as voltas que faltam e `reliability`:

* `indicative` (até 80% da corrida): no backtest o Monte Carlo não se distinguiu do comparador;
* `experimental` (a partir de 80%): no backtest o Monte Carlo foi PIOR que o comparador.

Nenhum dos dois valores quer dizer "confiável". A ressalva do viés de seleção dos alertas de
undercut (só as paradas que aconteceram foram medidas; a precisão é otimista) fica na DOCUMENTAÇÃO
do campo (`app/schemas.py`), não no payload.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from strategy.montecarlo import RaceForecast, SimInput
from strategy.undercut import Pair, UndercutAssessment

SCHEMA_VERSION = 1

# A partir de 80% da corrida o backtest achou o Monte Carlo pior que "o líder atual vence" (as
# faixas 80%-90% e 90%-100%). O corte é `>=` e não `>` para coincidir com o limite da faixa medida:
# a volta exatamente em 80% já cai na faixa em que ele foi pior.
EXPERIMENTAL_FROM = 0.80
INDICATIVE = "indicative"
EXPERIMENTAL = "experimental"

BASELINE_NAME = "o líder atual vence"

VSC_CODES = ("6", "7")
RED_CODE = "5"

WARNING_TEXT = {
    "safety_car": (
        "Safety Car em curso: o simulador o modela, mas a duração que falta é sorteada de poucos "
        "eventos históricos (o SC depois da volta 10 vem de 3 eventos)."
    ),
    "virtual_safety_car": (
        "VSC em curso: o simulador não modela o VSC. A probabilidade ignora esse efeito."
    ),
    "red_flag": (
        "Bandeira vermelha: fora do simulador. A probabilidade não vale enquanto a corrida estiver "
        "parada nem logo depois."
    ),
}


def reliability(lap: int, total_laps: int) -> str:
    return EXPERIMENTAL if lap / total_laps >= EXPERIMENTAL_FROM else INDICATIVE


def _flagged(rows: Sequence, lap: int, codes: Sequence[str]) -> bool:
    """Metade dos carros que completaram a `lap` tem algum dos `codes` (a mesma regra com que o
    simulador reconhece o SC em curso: `montecarlo._sc_elapsed`)."""
    hits = [
        any(code in (r.track_status or "") for code in codes) for r in rows if r.lap_number == lap
    ]
    return bool(hits) and sum(hits) >= 0.5 * len(hits)


def track_warnings(rows: Sequence, lap: int, sc_elapsed: int) -> list[dict[str, str]]:
    """Os avisos de pista: SC (o simulador o modela), VSC e bandeira vermelha (ele não). `rows` são
    as voltas JÁ visíveis no instante do estado; olha a volta do líder (`lap`) e a anterior, que é
    a que quase todo o pelotão já completou nesse instante."""
    codes = []
    if sc_elapsed > 0:
        codes.append("safety_car")
    recent = (lap, lap - 1)
    if any(_flagged(rows, n, VSC_CODES) for n in recent):
        codes.append("virtual_safety_car")
    if any(_flagged(rows, n, (RED_CODE,)) for n in recent):
        codes.append("red_flag")
    return [{"code": code, "text": WARNING_TEXT[code]} for code in codes]


def _num(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _alert(a: UndercutAssessment) -> dict[str, Any]:
    old_a, new_a, old_r, new_r = a.compounds
    return {
        "attacker": a.attacker,
        "rival": a.rival,
        "probability": _num(a.probability),
        "margin_s": _num(a.margin, 3),
        "margin_sd_s": _num(a.margin_sd, 3),
        "horizon_k": a.horizon_k,
        "pit_loss_s": _num(a.pit_loss, 3),
        "attacker_compounds": [old_a, new_a],
        "rival_compounds": [old_r, new_r],
        "blocked_by": a.blocked_by,
    }


def build_payload(
    *,
    year: int,
    round_number: int,
    epoch: int,
    lap: int,
    total_laps: int,
    forecast: RaceForecast,
    sim_input: SimInput,
    leader: str | None,
    assessments: Sequence[UndercutAssessment],
    alerts: set[Pair],
    warnings: Sequence[Mapping[str, str]],
    cars_known: int,
    late_laps: int,
    duplicates: int,
    wait_s: float,
    computed_at_ms: int,
) -> dict[str, Any]:
    """O documento da volta `lap`. Tudo o que não é relógio nem duração é função só das voltas que
    já tinham chegado (o teste de embaralhamento depende disso)."""
    drivers = [
        {
            "driver": d.driver,
            "win": _num(d.win),
            "podium": _num(d.podium),
            "win_error": _num(d.win_error),
            "podium_error": _num(d.podium_error),
            "baseline_win": 1.0 if d.driver == leader else 0.0,
            "retire": _num(d.retire),
            "expected_stops": _num(d.expected_stops, 3),
            "source": d.source,
        }
        for d in sorted(forecast.drivers, key=lambda d: (-d.win, d.driver))
    ]
    active = sorted(
        (a for a in assessments if (a.attacker, a.rival) in alerts),
        key=lambda a: (-(a.probability or 0.0), a.attacker),
    )
    return {
        "v": SCHEMA_VERSION,
        "type": "strategy",
        "year": year,
        "round": round_number,
        "epoch": epoch,
        "lap": lap,
        "total_laps": total_laps,
        "laps_left": total_laps - lap,
        "progress": _num(lap / total_laps, 4),
        "as_of_ms": sim_input.t_ms,
        "reliability": reliability(lap, total_laps),
        "baseline": {"name": BASELINE_NAME, "leader": leader},
        "forecast": {
            "n_sims": forecast.n_sims,
            "variant": forecast.variant,
            "sc_probability": _num(forecast.sc_probability),
            "without_model": list(forecast.without_model),
            "drivers": drivers,
        },
        "alerts": [_alert(a) for a in active],
        "warnings": [dict(w) for w in warnings],
        "inputs": {
            "cars_reported": len(sim_input.drivers) + len(sim_input.without_model),
            "cars_known": cars_known,
            "late_laps": late_laps,
            "duplicates": duplicates,
            "wait_s": wait_s,
            "constants": sim_input.sim_priors.source,
        },
        "computed_at_ms": computed_at_ms,
        "duration_s": _num(forecast.duration_s, 3),
    }


def dumps(payload: Mapping[str, Any]) -> str:
    """JSON compacto e estável (chaves na ordem de construção). `allow_nan=False`: um NaN que
    escapasse do modelo falha aqui, e não no navegador de quem lê."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


# Campos que mudam entre duas execuções do MESMO cálculo (relógio e duração): o que se compara em
# um teste de reprodutibilidade é o resto.
VOLATILE = ("computed_at_ms", "duration_s")


def stable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in VOLATILE}
