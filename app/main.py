from fastapi import FastAPI

app = FastAPI(
    title="F1 Race Engineer",
    description="API sobre dados de corridas de F1 ingeridos via FastF1.",
    version="0.1.0",
)


@app.get("/health", tags=["infra"])
def health() -> dict[str, str]:
    # Liveness: indica apenas que o processo está de pé e respondendo.
    # Usado pelo healthcheck do Docker Compose.
    return {"status": "ok"}
