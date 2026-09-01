"""Sweep — FastAPI backend for C++ terminal UI.

Pure API server. The C++ UI connects to this backend for data.
No web frontend, no templates, no static files.

    uvicorn app.main:app --reload --port 8787
    python -m app.main
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .api.routes import router

logger = logging.getLogger("sweep")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    cpp_status = "available" if _check_cpp() else "not compiled (pure Python mode)"
    logger.info(f"[Sweep] C++ engine: {cpp_status}")
    logger.info(f"[Sweep] Search providers: {settings.search_providers_configured}")
    yield


def _check_cpp() -> bool:
    try:
        import sweep_engine
        return True
    except ImportError:
        return False


app = FastAPI(
    title="Sweep API",
    description="Python/C++ web intelligence backend",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

settings = get_settings()
cors_origins = settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    torch_ok = False
    torch_version = ""
    try:
        import torch
        torch_ok = True
        torch_version = torch.__version__
    except ImportError:
        pass
    tf_ok = False
    tf_version = ""
    try:
        import tensorflow as tf
        tf_ok = True
        tf_version = tf.__version__
    except ImportError:
        pass
    return {
        "status": "ok",
        "service": "Sweep",
        "version": "2.0.0",
        "python": sys.version.split()[0],
        "cpp": _check_cpp(),
        "pytorch": torch_ok,
        "pytorch_version": torch_version,
        "tensorflow": tf_ok,
        "tensorflow_version": tf_version,
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
