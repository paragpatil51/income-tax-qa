"""FastAPI application: REST API under /api, and the static frontend at /.

Run from backend/:  uvicorn taxqa.api.main:app --port 8000   then open http://localhost:8000
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from taxqa.api import routes
from taxqa.core.config import FRONTEND_DIR, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("taxqa")


def _warmup() -> None:
    try:
        log.info("Loading index and models in the background ...")
        routes.get_pipeline().warmup()
        log.info("Ready to answer questions.")
    except routes.PipelineUnavailable as e:
        log.warning("Not ready: %s", e)
    except Exception:
        log.exception("Warm-up failed; models will load on the first question")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm up in a thread so the server (and /api/health) is reachable immediately.
    if settings.warmup:
        threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    yield


app = FastAPI(title="Income-tax Act QA", version="0.2.0", lifespan=lifespan)

# Allows the frontend to be served separately (e.g. `python -m http.server` in frontend/).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(routes.router)


@app.middleware("http")
async def _revalidate_frontend(request: Request, call_next):
    # Browsers otherwise keep an old stylesheet beside a new page after a redesign;
    # no-cache still serves 304s from the ETag, so this costs a round trip, not a download.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _error(status: int, kind: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": kind, "message": message, **extra}})


@app.exception_handler(routes.PipelineUnavailable)
async def _not_ready(_: Request, exc: routes.PipelineUnavailable) -> JSONResponse:
    return _error(503, "index_not_ready", str(exc))


@app.exception_handler(RequestValidationError)
async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    first = errors[0].get("msg", "invalid request") if errors else "invalid request"
    return _error(422, "validation_error", str(first).removeprefix("Value error, "),
                  details=[{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in errors])


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error")
    return _error(500, "internal_error", f"{type(exc).__name__}: {exc}")


if (FRONTEND_DIR / "index.html").exists():
    if (FRONTEND_DIR / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
