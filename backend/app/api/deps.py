"""Shared helpers for API routes."""
from __future__ import annotations

import json

from sqlmodel import select

from app.core.errors import BowlIQError
from app.models.db import Analysis, Feedback, session


def load_result(analysis_id: str, owner: str) -> dict:
    """The stored AnalysisResult, only if `owner` owns it (404 otherwise, so ids can't be probed)."""
    with session() as s:
        row = s.exec(select(Analysis).where(Analysis.id == analysis_id, Analysis.owner_id == owner)).first()
        if row is None:
            raise BowlIQError("NOT_FOUND", "That analysis doesn't exist (it may have been deleted).",
                              "Open My deliveries and choose another delivery.", 404)
        result = json.loads(row.result_json)
        fb = s.exec(select(Feedback).where(Feedback.analysis_id == analysis_id)).all()
    result["feedback"] = {str(f.finding_rank): f.helpful for f in fb}
    return result
