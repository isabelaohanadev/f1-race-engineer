"""As constantes que vêm dos dados (`Priors` do ritmo e da perda no pit, `SimPriors` do SC, dos
abandonos e das trocas de composto), lidas do banco, SEM a corrida que está sendo reproduzida.

Por que sem a corrida: o consumidor reproduz corridas que já estão no banco. Se as constantes a
incluíssem, o replay "saberia" coisas dela que nenhum sistema ao vivo saberia (o backtest as tira da
mesma forma: `leave_one_out`). Para uma corrida que não está no banco, ou fora da população seca
(chuva), nada sai e as constantes são as de todas as corridas secas.

Não são "só as corridas anteriores no calendário" (`earlier_only`): o backtest usou as outras 20, e
a configuração medida é a que se reproduz aqui. É uma diferença em relação ao que um sistema de
verdade, numa corrida nova, saberia, e está declarada no README.

As leituras do banco levam ~5 s (medido: 21 corridas) e acontecem UMA vez por processo, numa thread.
"""

import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from db.models import Race
from strategy.history import RaceHistory, leave_one_out_history, load_race_histories
from strategy.laps import race_label
from strategy.live import Constants
from strategy.priors import RaceStats, leave_one_out
from strategy.race_stats import load_race_stats

log = logging.getLogger(__name__)


class ConstantsProvider:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._population: tuple[list[RaceStats], list[RaceHistory]] | None = None
        self._lock = asyncio.Lock()
        self._cache: dict[tuple[int, int], Constants] = {}

    def _load_population(self) -> tuple[list[RaceStats], list[RaceHistory]]:
        with self._session_factory() as session:
            return load_race_stats(session), load_race_histories(session)

    def _label(self, year: int, round_number: int) -> str:
        with self._session_factory() as session:
            name = session.scalar(
                select(Race.name).where(Race.year == year, Race.round == round_number)
            )
        return "" if name is None else race_label(year, round_number, name)

    async def for_race(self, year: int, round_number: int) -> Constants:
        key = (year, round_number)
        async with self._lock:  # uma carga por vez, e só uma por processo
            if key in self._cache:
                return self._cache[key]
            if self._population is None:
                self._population = await asyncio.to_thread(self._load_population)
                log.info("constantes lidas do banco: %d corridas secas", len(self._population[0]))
            stats, histories = self._population
            label = await asyncio.to_thread(self._label, year, round_number)
            constants = Constants(
                leave_one_out(stats, label), leave_one_out_history(histories, label)
            )
            self._cache[key] = constants
            return constants
