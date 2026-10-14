from fastapi import FastAPI

from app.routers import health, races

app = FastAPI(
    title="F1 Race Engineer",
    description="API sobre dados de corridas de F1 ingeridos via FastF1.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(races.router)
