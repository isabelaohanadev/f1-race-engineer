"""Loader: baixa uma corrida pelo FastF1 e a converte em RaceData (tipos nativos do Python).

Tudo que é "peculiaridade do FastF1" fica ISOLADO aqui. O que descobrimos inspecionando o GP do
Brasil 2024 (e que explica várias linhas abaixo):

* Os datetimes do FastF1 vêm SEM fuso, mas são UTC: `session.date` é UTC por documentação e a
  largada real (t0_date + session_start_time = 15:49) é coerente com ele (agendada 15:30 UTC).
* `Date - t0_date == SessionTime` nos dados de posição; já `LapStartDate` NÃO serve para o fim da
  volta (com bandeira vermelha a volta seguinte só "começa" até ~23 min depois). Por isso o
  instante de término de uma volta é `t0_date + Time`.
* ~14% das amostras de posição são (0, 0): placeholder de "sem sinal de GPS" (antes do carro
  transmitir e também no meio da corrida). Não são posições reais e teletransportariam o carro
  para a origem do mapa, então são descartadas.
* NaN aparece em LapTime (voltas de largada/box), Position, Stint e TyreLife; e inteiros vêm como
  float. Tudo é normalizado para int | None. Nas posições, amostras com data ou coordenada nula
  são descartadas (não há como gravá-las).
* (Fase 3) `t0_date` só existe com a telemetria carregada (o FastF1 o deduz das amostras de posição
  e de carro). Por isso o `--laps-only` não tem horário absoluto das voltas, só o relógio da sessão.
* (Fase 3) Entrada e saída dos boxes vêm como INSTANTES (PitInTime/PitOutTime, nulos nas demais
  voltas) e guardamos só o fato (bool). Metade dessas voltas TEM tempo de volta (no Brasil: 17 das
  36 de entrada e 16 das 35 de saída), então "volta sem tempo" não identifica volta de box.
  TrackStatus é uma string de códigos ("1", "12", "124"...) com tudo o que ocorreu na volta.
"""

import logging
import os
from datetime import UTC, datetime

import fastf1
import pandas as pd

from ingestion.dto import EntryInfo, LapInfo, PositionTrack, RaceData, RaceInfo

logger = logging.getLogger(__name__)

SESSION_TYPE = "R"  # só a corrida (não sprint, qualy, treinos)


def load_race(year: int, round_number: int, cache_dir: str, *, laps_only: bool = False) -> RaceData:
    """Baixa (ou lê do cache em disco) e converte a corrida. Lento na 1ª vez: dezenas de MB.

    `laps_only=True` pula a telemetria de posição (a parte pesada: ~800 mil amostras por corrida,
    ~75 MB no banco) e devolve `tracks` vazio. Serve a quem só precisa de voltas e resultado, como a
    estratégia da Fase 3 (medido: ~7 a 14 s por corrida, contra minutos). A corrida fica SEM
    trajetória, então o replay dela não terá eventos de posição.
    """
    # O FastF1 exige que a pasta exista. O cache evita rebaixar tudo a cada execução.
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, round_number, SESSION_TYPE)
    # telemetry=True traz posição (pos_data). O FastF1 não permite carregar só a posição.
    session.load(laps=True, telemetry=not laps_only, weather=False, messages=False)

    # `t0_date` (o zero que converte o relógio da sessão em horário UTC) o FastF1 calcula A PARTIR
    # da telemetria: sem ela, ler `session.t0_date` levanta DataNotLoadedError. Em `laps_only` não
    # há horário absoluto; fica só o relógio da sessão (`session_time_ms`), que não precisa dele.
    t0 = None if laps_only else session.t0_date
    if not laps_only and t0 is None:
        raise RuntimeError(f"FastF1 não conseguiu determinar t0_date de {year} R{round_number}")

    entries = _build_entries(session.results)
    code_by_number = {e.driver_number: e.driver_code for e in entries}

    return RaceData(
        race=_build_race(session, year, round_number),
        entries=entries,
        laps=_build_laps(session.laps, t0),
        # Sem telemetria carregada, `session.pos_data` levanta DataNotLoadedError: nem tocamos nele.
        tracks=[] if laps_only else _build_tracks(session.pos_data, code_by_number),
        laps_only=laps_only,
    )


# ---------- conversões (funções puras: fáceis de testar com DataFrames pequenos) ----------


def _int_or_none(value: object) -> int | None:
    return None if pd.isna(value) else int(value)  # type: ignore[arg-type]


def _str_or_none(value: object) -> str | None:
    return None if pd.isna(value) or value == "" else str(value)


def _to_utc(ts: pd.Timestamp) -> datetime:
    # Timestamp "ingênuo" do FastF1 -> datetime com fuso UTC (ver nota no topo do módulo).
    return ts.to_pydatetime().replace(tzinfo=UTC)


def _build_race(session: fastf1.core.Session, year: int, round_number: int) -> RaceInfo:
    event = session.event
    laps = session.laps
    total_laps = getattr(session, "total_laps", None)
    if total_laps is None and len(laps):
        total_laps = laps["LapNumber"].max()
    return RaceInfo(
        year=year,
        round=round_number,
        name=str(event["EventName"]),
        location=_str_or_none(event["Location"]),
        country=_str_or_none(event["Country"]),
        starts_at=_to_utc(session.date),
        total_laps=_int_or_none(total_laps),
    )


def _build_entries(results: pd.DataFrame) -> list[EntryInfo]:
    entries = []
    for r in results.itertuples(index=False):
        color = _str_or_none(r.TeamColor)
        entries.append(
            EntryInfo(
                driver_code=str(r.Abbreviation),
                driver_number=int(r.DriverNumber),
                full_name=str(r.FullName),
                team_name=str(r.TeamName),
                # A API mistura caixa ("0093cc" e "3671C6"): normalizamos para maiúsculas.
                team_color=color.upper() if color and len(color) == 6 else None,
                grid_position=_int_or_none(r.GridPosition),
                finish_position=_int_or_none(r.Position),
                status=_str_or_none(r.Status),
            )
        )
    return entries


def _build_laps(laps: pd.DataFrame, t0: pd.Timestamp | None) -> list[LapInfo]:
    """`t0` nulo = sem telemetria (`laps_only`): `completed_at` sai nulo, o resto não muda."""
    out = []
    for lap in laps.itertuples(index=False):
        lap_time = None if pd.isna(lap.LapTime) else round(lap.LapTime.total_seconds() * 1000)
        completed_at = None if t0 is None or pd.isna(lap.Time) else _to_utc(t0 + lap.Time)
        session_time = None if pd.isna(lap.Time) else round(lap.Time.total_seconds() * 1000)
        out.append(
            LapInfo(
                driver_code=str(lap.Driver),
                lap_number=int(lap.LapNumber),
                lap_time_ms=lap_time,
                compound=_str_or_none(lap.Compound),
                tyre_life=_int_or_none(lap.TyreLife),
                stint=_int_or_none(lap.Stint),
                position=_int_or_none(lap.Position),
                completed_at=completed_at,
                # Presença do instante = a volta tem entrada/saída dos boxes. Sempre True/False
                # (nunca None): o FastF1 informa os dois para toda volta.
                pit_in=not pd.isna(lap.PitInTime),
                pit_out=not pd.isna(lap.PitOutTime),
                track_status=_str_or_none(lap.TrackStatus),
                session_time_ms=session_time,
            )
        )
    return out


def _build_tracks(
    pos_data: dict[str, pd.DataFrame], code_by_number: dict[int, str]
) -> list[PositionTrack]:
    tracks = []
    for number, df in pos_data.items():
        code = code_by_number.get(int(number))
        if code is None:
            # Carro sem inscrição nos resultados (ex.: veículos de apoio). Sem piloto, sem FK.
            logger.warning("Posições do carro #%s ignoradas: não consta nos resultados", number)
            continue

        # 1) Amostras inutilizáveis: sem instante (`time` é parte da PK e a ordenação perde o
        #    sentido) ou com coordenada nula (int(NaN) levanta erro, e x/y são NOT NULL no banco).
        #    Sem esta linha, UM valor nulo derrubava a ingestão da corrida inteira.
        df = df.dropna(subset=["Date", "X", "Y"])
        # 2) Placeholder (0, 0) = "sem sinal de GPS" (ver nota no topo do módulo).
        df = df[(df["X"] != 0) | (df["Y"] != 0)].sort_values("Date")
        # pandas 2.x: DatetimeIndex.to_pydatetime() devolve datetimes nativos (com UTC).
        times = pd.DatetimeIndex(df["Date"]).tz_localize("UTC").to_pydatetime().tolist()
        tracks.append(
            PositionTrack(
                driver_code=code,
                times=times,
                xs=df["X"].astype(int).tolist(),
                ys=df["Y"].astype(int).tolist(),
            )
        )
    return tracks
