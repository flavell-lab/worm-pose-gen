"""Recordings: frame thumbnails for the Recordings screen, and the file explorer of Add recording."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from . import get_state, query_flag
from ..state import AppState

router = APIRouter(prefix="/api/recordings")


@router.get("/thumbnail")
def thumbnail(path: str, frame: int = 0, scale: float = 0.25, app: AppState = Depends(get_state)) -> Response:
    return Response(content=app.thumbnail(path, frame, scale), media_type="image/png")


files_router = APIRouter(prefix="/api/files")


@files_router.get("")
def browse(path: str | None = None, all: str | None = None, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The file explorer: directories, HDF5 files and videos under ``path`` (every file with ``all=1``)."""

    return app.browse(path, query_flag(all))
