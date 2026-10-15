"""As constantes do consumidor: lidas do banco uma vez e SEM a corrida reproduzida."""

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from strategy.service import constants as module
from strategy.service.constants import ConstantsProvider
from tests.replay.conftest import seed_race


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


async def test_the_constants_leave_the_replayed_race_out_and_the_population_loads_once(
    engine, session_factory, monkeypatch
):
    seed_race(engine, positions=[], laps=[], total_laps=3, round_number=10)  # 2024 R10, nome "GP"
    loads, excluded = [], []
    monkeypatch.setattr(module, "load_race_stats", lambda session: loads.append("stats") or ["S"])
    monkeypatch.setattr(
        module, "load_race_histories", lambda session: loads.append("hist") or ["H"]
    )
    monkeypatch.setattr(
        module, "leave_one_out", lambda stats, label: excluded.append(("priors", label)) or "P"
    )
    monkeypatch.setattr(
        module, "leave_one_out_history", lambda hist, label: excluded.append(("sim", label)) or "SP"
    )
    provider = ConstantsProvider(session_factory)

    first = await provider.for_race(2024, 10)
    again = await provider.for_race(2024, 10)
    unknown = await provider.for_race(2024, 3)  # não está no banco: nada a excluir

    assert (first.priors, first.sim_priors) == ("P", "SP") and again is first  # em cache
    assert loads == ["stats", "hist"]  # a população foi lida uma vez, não uma por corrida
    assert excluded == [
        ("priors", "2024 R10 GP"), ("sim", "2024 R10 GP"), ("priors", ""), ("sim", "")
    ]  # fmt: skip
    assert unknown.priors == "P"


async def test_an_empty_database_gives_the_default_constants_instead_of_failing(session_factory):
    constants = await ConstantsProvider(session_factory).for_race(2024, 10)

    assert constants.priors is not None and constants.sim_priors is not None
