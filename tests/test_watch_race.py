"""As regras do cliente de referência (scripts/watch_race.py): são o CONTRATO do protocolo que o
README ensina. Se uma regra mudar, quem copiou o exemplo precisa saber (e este teste quebra)."""

from scripts.watch_race import RaceView


def position(driver: str, t: int, *, epoch: int = 1) -> dict:
    return {"type": "position", "epoch": epoch, "driver": driver, "x": t, "y": -t, "t": t}


def lap(driver: str, n: int, t: int, *, epoch: int = 1, **extra) -> dict:
    base = {"lap_time_ms": 90_000, "compound": "SOFT", "position": 1}
    return (
        {"type": "lap_completed", "epoch": epoch, "driver": driver, "lap": n, "t": t} | base | extra
    )


def status(state: str, lap_number: int, t: int = 0, *, epoch: int = 1) -> dict:
    return {
        "type": "race_status",
        "epoch": epoch,
        "state": state,
        "lap": lap_number,
        "t": t,
        "total_laps": 69,
    }


def test_a_duplicate_or_older_position_does_not_overwrite_a_newer_one():
    view = RaceView()
    view.apply_event(position("VER", 200))

    view.apply_event(position("VER", 200))  # repetido (o catch-up repete o que o snapshot tinha)
    view.apply_event(position("VER", 100))  # atrasado

    assert view.cars["VER"] == {"x": 200, "y": -200, "pos_t": 200}


def test_an_event_from_an_older_replay_is_ignored_and_a_newer_epoch_resets_everything():
    view = RaceView()
    view.apply_snapshot({"epoch": 2, "race": {}, "cars": {"VER": {"x": 5, "pos_t": 5}}})

    view.apply_event(position("VER", 999_999, epoch=1))  # replay antigo, mesmo com t gigante
    assert view.cars["VER"]["x"] == 5

    view.apply_event(position("NOR", 7, epoch=3))  # replay NOVO: descarta o estado anterior
    assert (view.epoch, set(view.cars)) == (3, {"NOR"})


def test_a_lap_without_a_value_does_not_inherit_it_from_the_previous_lap():
    view = RaceView()
    view.apply_event(lap("VER", 1, 1_000, compound="SOFT", lap_time_ms=90_000))

    view.apply_event(lap("VER", 2, 2_000, compound=None, lap_time_ms=None))

    assert "compound" not in view.cars["VER"] and "lap_time_ms" not in view.cars["VER"]
    assert view.cars["VER"]["lap"] == 2


def test_position_and_lap_versions_are_independent():
    view = RaceView()
    view.apply_event(lap("VER", 1, 9_000))  # uma volta "nova" primeiro...

    view.apply_event(position("VER", 5_000))  # ...não pode descartar a 1ª posição, mais antiga

    assert view.cars["VER"]["x"] == 5_000


def test_the_race_status_only_moves_forward_in_lap_and_state():
    view = RaceView()
    view.apply_event(status("started", 1))
    view.apply_event(status("running", 3))
    view.apply_event(status("running", 2))  # atrasado: volta menor
    view.apply_event(status("started", 3))  # atrasado: estado anterior na mesma volta

    assert view.race == {"state": "running", "lap": 3, "total_laps": 69}

    view.apply_event(status("finished", 3))  # mesma volta, estado posterior

    assert view.finished


def test_classification_orders_by_laps_then_by_who_crossed_first_and_unlapped_last():
    view = RaceView()
    for event in (
        lap("VER", 5, 900),
        lap("NOR", 5, 800),  # mesma volta, cruzou antes
        lap("HAM", 6, 1_000),  # mais voltas lidera, mesmo cruzando depois
        position("ALB", 10),  # ainda sem volta completa
    ):
        view.apply_event(event)

    assert view.classification == ["HAM", "NOR", "VER", "ALB"]
