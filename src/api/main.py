"""FastAPI entrypoint for the NYC Restaurant Intelligence Platform backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.datasets import router as datasets_router
from .routers.health import router as health_router
from .routers.recommendations import router as recommendations_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # startup: nothing needed yet
    yield
    # shutdown: nothing needed yet


app = FastAPI(
    title="NYC Restaurant Intelligence Platform API",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(datasets_router)
app.include_router(recommendations_router)
