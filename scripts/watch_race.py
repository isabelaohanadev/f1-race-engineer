#!/usr/bin/env python
"""Cliente de REFERÊNCIA do WebSocket de corrida: conecta, aplica o snapshot e os eventos e mostra a
corrida ao vivo no terminal.

    python scripts/watch_race.py --year 2024 --round 21
    python scripts/watch_race.py --year 2024 --round 21 --events     # imprime cada evento

Precisa só do pacote `websockets` (já vem com `uvicorn[standard]`; ou `pip install websockets`).

Este arquivo também DOCUMENTA, em código, o que um cliente precisa fazer (a classe `RaceView`), e a
suíte de testes o reutiliza: o exemplo que você copia é o mesmo que os testes validam. As regras:

* O 1º frame é o `snapshot`. Depois chegam eventos, que PODEM repetir o que o snapshot já refletia
  (é o preço de não ter buracos): aplicá-los de novo é inofensivo desde que o cliente use as
  versões abaixo, em vez de simplesmente sobrescrever.
* `epoch` identifica o replay. Um evento com epoch MAIOR que o seu é de um replay novo: descarte o
  estado e recomece. Um com epoch MENOR é de um replay antigo: ignore.
* Por piloto, uma POSIÇÃO só vale se `t` > `pos_t` do que você já tem; uma VOLTA, se `t` > `lap_t`.
* O status da corrida só avança em (volta, estado): started < running < finished.
* A classificação é derivada: mais voltas completas primeiro; empatando, quem cruzou a linha antes.
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

STATE_RANK = {"started": 0, "running": 1, "finished": 2}
CLOSE_TOO_SLOW = (
    1013  # o servidor desconecta quem não acompanha: reconecte e receba um snapshot novo
)


class RaceView:
    """O estado da corrida como o cliente o enxerga, reconstruído do snapshot e dos eventos."""

    def __init__(self) -> None:
        self.epoch = 0
        self.race: dict[str, Any] = {}
        self.cars: dict[str, dict[str, Any]] = {}

    def apply_snapshot(self, message: dict[str, Any]) -> None:
        self.epoch = message["epoch"]
        self.race = dict(message["race"])
        self.cars = {code: dict(car) for code, car in message["cars"].items()}

    def apply_event(self, message: dict[str, Any]) -> None:
        if message["epoch"] < self.epoch:
            return  # evento de um replay antigo
        if message["epoch"] > self.epoch:
            self.epoch, self.race, self.cars = message["epoch"], {}, {}  # replay novo
        match message["type"]:
            case "position":
                car = self.cars.setdefault(message["driver"], {})
                if car.get("pos_t", -1) < message["t"]:
                    car |= {"x": message["x"], "y": message["y"], "pos_t": message["t"]}
            case "lap_completed":
                car = self.cars.setdefault(message["driver"], {})
                if car.get("lap_t", -1) < message["t"]:
                    car |= {"lap": message["lap"], "lap_t": message["t"]}
                    for name in ("lap_time_ms", "compound", "position"):
                        if message[name] is None:
                            car.pop(name, None)  # volta sem esse dado: não herda a anterior
                        else:
                            car[name] = message[name]
            case "race_status":
                current = (self.race.get("lap", 0), STATE_RANK.get(self.race.get("state", ""), -1))
                if (message["lap"], STATE_RANK[message["state"]]) > current:
                    self.race = {"state": message["state"], "lap": message["lap"]}
                    if message["total_laps"] is not None:
                        self.race["total_laps"] = message["total_laps"]

    @property
    def finished(self) -> bool:
        return self.race.get("state") == "finished"

    @property
    def classification(self) -> list[str]:
        """Mais voltas primeiro; empatando, quem cruzou antes; sem volta nenhuma, por último."""
        with_laps = [(code, car) for code, car in self.cars.items() if "lap" in car]
        without = sorted(code for code, car in self.cars.items() if "lap" not in car)
        with_laps.sort(key=lambda item: (-item[1]["lap"], item[1].get("lap_t", 0), item[0]))
        return [code for code, _ in with_laps] + without


def describe(view: RaceView) -> str:
    lap, total = view.race.get("lap", "-"), view.race.get("total_laps", "?")
    top = " ".join(view.classification[:5]) or "-"
    return (
        f"volta {lap}/{total} {view.race.get('state', '-')} | {len(view.cars)} carros | top5: {top}"
    )


async def watch(url: str, *, show_events: bool, duration: float | None) -> int:
    view, events, started = RaceView(), 0, time.monotonic()
    last_report, last_count = started, 0
    try:
        async with websockets.connect(url, max_size=None) as ws:
            print(f"conectado: {url}")
            snapshot = json.loads(await ws.recv())
            view.apply_snapshot(snapshot)
            replay = snapshot.get("replay")
            print(f"snapshot: epoch {snapshot['epoch']} | {describe(view)}")
            print(
                f"replay: {replay['state']} a {replay['speed']:g}x" if replay else "replay: nenhum"
            )
            async for raw in ws:
                message = json.loads(raw)
                view.apply_event(message)
                events += 1
                if show_events:
                    print(raw)
                now = time.monotonic()
                if now - last_report >= 1.0:
                    rate = (events - last_count) / (now - last_report)
                    print(f"[{now - started:5.0f}s] {rate:7,.0f} eventos/s | {describe(view)}")
                    last_report, last_count = now, events
                if view.finished:
                    print("corrida terminada.")
                    return 0
                if duration is not None and now - started >= duration:
                    return 0
    except ConnectionClosed as closed:
        if closed.rcvd and closed.rcvd.code == CLOSE_TOO_SLOW:
            print(
                "o servidor desconectou este cliente por ser lento (1013): reconecte.",
                file=sys.stderr,
            )
        else:
            print(f"conexão encerrada: {closed}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"não consegui conectar em {url}: {error}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--round", type=int, default=21)
    parser.add_argument("--host", default="localhost:8000", help="host:porta da API")
    parser.add_argument("--events", action="store_true", help="imprime cada evento (muito volume!)")
    parser.add_argument("--duration", type=float, help="encerra depois de N segundos")
    args = parser.parse_args()
    url = f"ws://{args.host}/ws/races/{args.year}/{args.round}"
    try:
        return asyncio.run(watch(url, show_events=args.events, duration=args.duration))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
