from typing import Any, Optional

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_ctx

logger = get_logger("predictguard.errors")


class AppException(Exception):
    """Base application exception with error code and status code."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details


class ModelNotLoadedException(AppException):
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' is not loaded or unavailable.",
            error_code="MODEL_NOT_AVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class ImageValidationException(AppException):
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST, details: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code="INVALID_IMAGE",
            status_code=status_code,
            details=details,
        )


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or request_id_ctx.get() or "unknown"


def make_error_response(
    status_code: int,
    error_code: str,
    message: str,
    request_id: str,
    details: Optional[Any] = None,
) -> JSONResponse:
    content = {
        "error_code": error_code,
        "message": message,
        "request_id": request_id,
        "detail": message,  # for backward compatibility with FastAPI default 'detail' field
    }
    if details is not None:
        content["details"] = details
    return JSONResponse(status_code=status_code, content=content)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    request_id = _get_request_id(request)
    logger.warning(
        "Application exception: %s (code=%s, status=%d)",
        exc.message,
        exc.error_code,
        exc.status_code,
        extra={"request_id": request_id},
    )
    return make_error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        request_id=request_id,
        details=exc.details,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = _get_request_id(request)
    error_code = "HTTP_ERROR"
    if exc.status_code == status.HTTP_404_NOT_FOUND:
        error_code = "NOT_FOUND"
    elif exc.status_code == status.HTTP_400_BAD_REQUEST:
        error_code = "BAD_REQUEST"
    elif exc.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE:
        error_code = "PAYLOAD_TOO_LARGE"
    elif exc.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE:
        error_code = "UNSUPPORTED_MEDIA_TYPE"
    elif exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        error_code = "SERVICE_UNAVAILABLE"

    message = str(exc.detail) if exc.detail else "An HTTP error occurred."
    return make_error_response(
        status_code=exc.status_code,
        error_code=error_code,
        message=message,
        request_id=request_id,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = _get_request_id(request)
    # Simplify validation error messages for clean readability
    errors = exc.errors()
    msg_parts = []
    for err in errors:
        loc = " -> ".join(str(part) for part in err.get("loc", []))
        msg_parts.append(f"{loc}: {err.get('msg')}")
    message = "Request validation failed: " + "; ".join(msg_parts) if msg_parts else "Invalid request data."

    return make_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="VALIDATION_ERROR",
        message=message,
        request_id=request_id,
        details=errors,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = _get_request_id(request)
    logger.exception("Unhandled server exception: %s", exc, extra={"request_id": request_id})
    return make_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred.",
        request_id=request_id,
    )
