"""OpenRouter-style error envelope: ``{"error": {"code", "message", "metadata"}}``.

The HTTP status always equals ``error.code``, as on OpenRouter, so OpenAI and
OpenRouter SDKs surface our failures the same way they surface theirs.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(
        self,
        code: int,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.metadata = metadata
        self.headers = headers


def error_response(
    code: int,
    message: str,
    metadata: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if metadata:
        error["metadata"] = metadata
    return JSONResponse({"error": error}, status_code=code, headers=headers)


def _log(request: Request, code: int, message: str) -> None:
    """One line per error response, so the access log's bare "400" has a reason."""
    if code == 404 and request.url.path.startswith("/api/v1/models/"):
        level = logging.INFO  # a client probing for a model is not a problem
    else:
        level = logging.ERROR if code >= 500 else logging.WARNING
    logger.log(level, "%d %s %s: %s", code, request.method, request.url.path, message)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        _log(request, exc.code, exc.message)
        return error_response(exc.code, exc.message, exc.metadata, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        _log(request, exc.status_code, str(exc.detail))
        return error_response(exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        where = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = first.get("msg", "Invalid request")
        _log(request, 400, f"{where}: {message}" if where else message)
        return error_response(
            400,
            f"{where}: {message}" if where else message,
            {"errors": [{k: e.get(k) for k in ("loc", "msg", "type")} for e in errors]},
        )
