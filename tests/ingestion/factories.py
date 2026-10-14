"""Monta DataFrames com o MESMO formato (colunas e dtypes) que o FastF1 devolve.

Os dtypes abaixo foram copiados de uma sessão real (GP do Brasil 2024), não inventados: um
DataFrame de teste com dtypes diferentes poderia passar mesmo que o loader quebrasse com o dado de
verdade. Cada fábrica só traz as colunas que o loader lê; qualquer chave omitida vira nulo.

Convenções do FastF1 que os testes precisam reproduzir:
* tempos são `timedelta64[ns]`; instantes são `datetime64[ns]` SEM fuso (mas em UTC);
* "inteiros" (LapNumber, Stint, TyreLife, Position...) vêm como float64, porque aceitam NaN;
* valores ausentes aparecem como NaT (tempos), NaN (números) ou None/NaN (textos).
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

RESULTS_DTYPES = {
    "DriverNumber": "object",
    "Abbreviation": "object",
    "FullName": "object",
    "TeamName": "object",
    "TeamColor": "object",
    "GridPosition": "float64",
    "Position": "float64",
    "Status": "object",
}

LAPS_DTYPES = {
    "Time": "timedelta64[ns]",  # tempo de SESSÃO no fim da volta (não é a duração da volta!)
    "Driver": "object",
    "LapTime": "timedelta64[ns]",  # duração da volta
    "LapNumber": "float64",
    "Stint": "float64",
    "Compound": "object",
    "TyreLife": "float64",
    "Position": "float64",
}

POS_DTYPES = {"Date": "datetime64[ns]", "X": "float64", "Y": "float64"}


def _column(values: list[Any], dtype: str) -> pd.Series:
    # Cada coluna é construída com o conversor do SEU tipo. Um `astype` em cima de uma coluna
    # `object` falha em casos de borda: uma coluna só com NaT é inferida como datetime e não
    # converte para timedelta. O `astype` final fixa a resolução em ns (o que o FastF1 usa),
    # independente do padrão da versão do pandas instalada.
    raw = pd.Series(values, dtype="object")
    if dtype.startswith("timedelta64"):
        return pd.to_timedelta(raw).astype(dtype)
    if dtype.startswith("datetime64"):
        return pd.to_datetime(raw).astype(dtype)
    return raw.astype(dtype)


def _frame(rows: list[dict[str, Any]], dtypes: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {col: _column([row.get(col) for row in rows], dtype) for col, dtype in dtypes.items()}
    )


def make_results(*rows: dict[str, Any]) -> pd.DataFrame:
    return _frame(list(rows), RESULTS_DTYPES)


def make_laps(*rows: dict[str, Any]) -> pd.DataFrame:
    return _frame(list(rows), LAPS_DTYPES)


def make_positions(*rows: tuple[str, float, float]) -> pd.DataFrame:
    """Cada linha é (data_utc_naive, x, y). Use np.nan / None / "NaT" para nulos."""
    return _frame([{"Date": d, "X": x, "Y": y} for d, x, y in rows], POS_DTYPES)


DEFAULT_T0 = pd.Timestamp("2024-11-03 14:38:02.746")  # t0_date real do GP do Brasil 2024
DEFAULT_DATE = pd.Timestamp("2024-11-03 15:30:00")  # session.date (agendada, UTC)


def make_session(
    *,
    results: pd.DataFrame,
    laps: pd.DataFrame,
    pos_data: dict[str, pd.DataFrame],
    t0_date: pd.Timestamp | None = DEFAULT_T0,
    total_laps: int | None = 69,
    event: dict[str, Any] | None = None,
    date: pd.Timestamp = DEFAULT_DATE,
) -> SimpleNamespace:
    """Substituto de `fastf1.core.Session` com só o que o loader usa."""
    event = event or {
        "EventName": "São Paulo Grand Prix",
        "Location": "São Paulo",
        "Country": "Brazil",
    }
    return SimpleNamespace(
        event=pd.Series(event),  # no FastF1, `event` também é indexável por nome (event["X"])
        date=date,
        t0_date=t0_date,
        total_laps=total_laps,
        results=results,
        laps=laps,
        pos_data=pos_data,
    )


NAN = np.nan
