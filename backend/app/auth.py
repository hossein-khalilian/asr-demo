"""Bearer API keys, checked the way OpenRouter checks them."""

import hmac

from fastapi import Depends, Request

from app.errors import APIError
from app.settings import Settings, get_settings


def require_api_key(request: Request, settings: Settings = Depends(get_settings)) -> None:
    if not settings.api_keys:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise APIError(401, "Missing Authorization: Bearer <key> header.")
    # Compare against every key so the timing does not reveal which one matched.
    matched = False
    for key in settings.api_keys:
        matched |= hmac.compare_digest(token.encode(), key.encode())
    if not matched:
        raise APIError(401, "Invalid API key.")
