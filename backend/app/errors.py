"""OpenRouter-style error envelope: ``{"error": {"code", "message", "metadata"}}``.

The HTTP status always equals ``error.code``, as on OpenRouter, so OpenAI and
OpenRouter SDKs surface our failures the same way they surface theirs.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


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


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return error_response(exc.code, exc.message, exc.metadata, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        where = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = first.get("msg", "Invalid request")
        return error_response(
            400,
            f"{where}: {message}" if where else message,
            {"errors": [{k: e.get(k) for k in ("loc", "msg", "type")} for e in errors]},
        )
