from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.deps import get_db
from app.schemas import ErrorResponse, LapOut, RaceLapsResponse, RaceOut
from db.models import Driver, Lap, Race, RaceEntry

router = APIRouter(prefix="/races", tags=["races"])


@router.get(
    "/{year}/{round}/laps",
    response_model=RaceLapsResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Corrida não ingerida, ou piloto que não participou dela.",
        }
    },
)
def get_race_laps(
    year: Annotated[int, Path(ge=1950, le=2100, description="Temporada, ex.: 2024")],
    round_number: Annotated[
        int, Path(alias="round", ge=1, le=50, description="Etapa do campeonato, ex.: 21")
    ],
    db: Annotated[Session, Depends(get_db)],
    driver: Annotated[
        str | None,
        Query(
            pattern=r"^[A-Za-z]{3}$",
            description="Código de 3 letras do piloto, ex.: VER (não diferencia maiúsculas).",
        ),
    ] = None,
) -> RaceLapsResponse:
    """Voltas da corrida (tempo, composto, idade do pneu, stint), ordenadas por piloto e volta.

    Sem paginação de propósito: uma corrida tem ~1.100 a 1.500 voltas (poucas centenas de KB).
    """
    race = db.scalar(select(Race).where(Race.year == year, Race.round == round_number))
    if race is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=(
                f"Corrida {year}/{round_number} não encontrada. Ingira-a com: "
                f"python -m ingestion --year {year} --round {round_number}"
            ),
        )

    stmt = (
        select(
            Driver.code.label("driver"),
            Lap.lap_number,
            Lap.lap_time_ms,
            Lap.compound,
            Lap.tyre_life,
            Lap.stint,
            Lap.position,
            Lap.completed_at,
        )
        .join(Driver, Driver.id == Lap.driver_id)
        .where(Lap.race_id == race.id)
        .order_by(Driver.code, Lap.lap_number)
    )

    code = driver.upper() if driver else None
    if code:
        # Quem participou DESTA corrida vem de race_entries (e não de laps): um piloto que não
        # largou (DNS) não tem voltas, mas existe na corrida, e isso é um 200 com lista vazia.
        # 404 é só para quem não consta na corrida, para não confundir "typo" com "sem voltas".
        entries = db.execute(
            select(Driver.id, Driver.code)
            .join(RaceEntry, RaceEntry.driver_id == Driver.id)
            .where(RaceEntry.race_id == race.id)
        ).all()
        driver_ids = {entry_code: driver_id for driver_id, entry_code in entries}
        if code not in driver_ids:
            available = ", ".join(sorted(driver_ids))
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Piloto '{code}' não participou da corrida {year}/{round_number}. "
                    f"Pilotos desta corrida: {available}."
                ),
            )
        stmt = stmt.where(Lap.driver_id == driver_ids[code])

    laps = [LapOut.model_validate(row) for row in db.execute(stmt).all()]
    return RaceLapsResponse(race=RaceOut.model_validate(race), driver=code, laps=laps)
