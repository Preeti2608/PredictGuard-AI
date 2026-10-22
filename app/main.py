import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.core.errors import (
    AppException,
    app_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.logging import get_logger, request_id_ctx, setup_logging
from app.core.rate_limit import limiter
from app.routers import health, image, models, sensors
from app.services.model_registry import get_model_registry

logger = get_logger("predictguard.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI application lifespan managing startup and shutdown phases."""
    settings = get_settings()
    setup_logging(debug=settings.debug)
    logger.info("Starting up %s (v%s) in %s mode...", settings.app_name, settings.app_version, settings.environment)

    # Load machine learning models once at startup into memory
    registry = get_model_registry()
    registry.load_all(settings)

    if registry.is_ready():
        logger.info("All required machine learning models loaded successfully.")
    else:
        logger.warning("One or more ML models failed to load. Readiness probe will fail until resolved.")

    yield

    logger.info("Shutting down %s...", settings.app_name)


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Application factory for PredictGuard AI."""
    settings = custom_settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Industrial AI Predictive Maintenance Platform: Surface Defect Classification & Telemetry Failure Forecasting",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Attach rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Attach custom exception handlers for uniform {error_code, message, request_id} responses
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Request-ID and Timing Middleware
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = req_id
        token = request_id_ctx.set(req_id)
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)

        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response

    # CORS configuration
    # Only configure if explicit origins provided; default is same-origin (no wildcard + credentials!)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
        )

    # Register Routers
    app.include_router(health.router)
    app.include_router(image.router)
    app.include_router(sensors.router)
    app.include_router(models.router)

    # Mount static assets
    frontend_dir = settings.base_dir / settings.frontend_dir
    static_dir = frontend_dir / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    sample_dir = settings.base_dir / settings.sample_images_dir
    if sample_dir.exists():
        app.mount("/sample_images", StaticFiles(directory=str(sample_dir)), name="sample_images")

    # Root route serves frontend or fallback API info
    @app.get("/", include_in_schema=False)
    def home_page() -> Response:
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return FileResponse(str(settings.base_dir / "index.html"))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    app_settings = get_settings()
    uvicorn.run("app.main:app", host=app_settings.host, port=app_settings.port, reload=app_settings.debug)
