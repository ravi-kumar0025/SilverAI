"""FastAPI application entry point for SilverAI."""

from __future__ import annotations

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import CORS_ORIGINS
from backend.core.config import DATA_MODE
from backend.routes.api import router
from backend.services.prediction_service import PredictionService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

@asynccontextmanager
async def lifespan(application: FastAPI):
    task = None
    if DATA_MODE == "live":
        async def poll() -> None:
            while True:
                await asyncio.to_thread(application.state.prediction_service.poll_live_logs)
                await asyncio.sleep(5)
        task = asyncio.create_task(poll())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


app = FastAPI(
    title="SilverAI Backend",
    version="1.0.0",
    description="REST API for the SilverAI prediction pipeline.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)
app.state.prediction_service = PredictionService()
app.include_router(router)
