"""Typed pipeline/API errors. Every error carries code, message and a fix."""
from __future__ import annotations


class BowlIQError(Exception):
    def __init__(self, code: str, message: str, fix: str, http_status: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix = fix
        self.http_status = http_status

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "fix": self.fix}


FILMING_FIX = (
    "Film side-on, square to the crease, about 10 m away at hip height, on a tripod, at 60 fps or more, "
    "with the whole body in view from the last run-up strides to the follow-through."
)
