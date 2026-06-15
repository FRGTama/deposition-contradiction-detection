from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analyze import analyze_depositions
from .models import AnalysisRequest


app = FastAPI(title="Deposition Contradiction Detector")
client_dist = Path(__file__).resolve().parent.parent / "dist" / "client"


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    try:
        result = await analyze_depositions(request.transcript1, request.transcript2)
        return result.model_dump(by_alias=True, exclude_none=True)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


if client_dist.exists():
    app.mount("/assets", StaticFiles(directory=client_dist / "assets"), name="assets")

    @app.get("/{path:path}")
    async def serve_client(path: str):
        requested_path = client_dist / path
        if path and requested_path.exists() and requested_path.is_file():
            return FileResponse(requested_path)
        return FileResponse(client_dist / "index.html")
