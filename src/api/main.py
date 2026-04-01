"""FastAPI entrypoint for the placeholder backend."""

from fastapi import FastAPI

from .routers.datasets import router as datasets_router
from .routers.health import router as health_router
from .routers.recommendations import router as recommendations_router

app = FastAPI(title="NYC Restaurant Intelligence Platform API", version="0.1.0")
app.include_router(health_router)
app.include_router(datasets_router)
app.include_router(recommendations_router)
