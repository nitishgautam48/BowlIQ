"""SQLModel tables, engine and versioned schema migrations (SQLite, PRAGMA user_version)."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, UniqueConstraint, event, text
from sqlmodel import Field, Session, SQLModel, create_engine

from app.core.config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(d: Optional[datetime]) -> Optional[str]:
    """ISO-8601 with an explicit UTC offset, whether SQLite hands back an aware or a naive datetime."""
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).isoformat(timespec="seconds")


def _ts() -> Column:
    return Column(DateTime(timezone=True), nullable=False, default=utcnow)


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    owner_id: str = Field(default="owner", index=True)
    status: str = Field(default="queued", index=True)
    stage: str = "Queued"
    progress: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_fix: Optional[str] = None
    params_json: str = "{}"
    worker: Optional[str] = None
    lease_until: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    created_at: datetime = Field(default_factory=utcnow, sa_column=_ts())
    updated_at: datetime = Field(default_factory=utcnow, sa_column=_ts())


class Analysis(SQLModel, table=True):
    id: str = Field(primary_key=True)
    owner_id: str = Field(default="owner", index=True)
    bowler_name: str = Field(index=True)
    created_at: datetime = Field(default_factory=utcnow, sa_column=_ts())
    ball_speed_kmh: Optional[float] = None
    quality: str = "Low"
    top_finding: Optional[str] = None
    has_front: bool = False
    metrics_json: str = "{}"
    result_json: str = "{}"


class Feedback(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("analysis_id", "finding_rank", name="uq_feedback_rank"),)
    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: str = Field(foreign_key="analysis.id", index=True, ondelete="CASCADE")
    finding_rank: int
    helpful: bool
    created_at: datetime = Field(default_factory=utcnow, sa_column=_ts())


# Each entry upgrades the schema by one version. Version 1 = the original (pre-audit) schema created by create_all.
MIGRATIONS: list[list[str]] = [
    # v2: ownership, job leases, FK + cascade + unique feedback, composite index
    [
        "ALTER TABLE job ADD COLUMN owner_id VARCHAR NOT NULL DEFAULT 'owner'",
        "ALTER TABLE job ADD COLUMN worker VARCHAR",
        "ALTER TABLE job ADD COLUMN lease_until DATETIME",
        "ALTER TABLE analysis ADD COLUMN owner_id VARCHAR NOT NULL DEFAULT 'owner'",
        "CREATE TABLE feedback_new (id INTEGER PRIMARY KEY, analysis_id VARCHAR NOT NULL "
        "REFERENCES analysis(id) ON DELETE CASCADE, finding_rank INTEGER NOT NULL, helpful BOOLEAN NOT NULL, "
        "created_at DATETIME NOT NULL, CONSTRAINT uq_feedback_rank UNIQUE (analysis_id, finding_rank))",
        "INSERT OR REPLACE INTO feedback_new (analysis_id, finding_rank, helpful, created_at) "
        "SELECT analysis_id, finding_rank, helpful, created_at FROM feedback "
        "WHERE analysis_id IN (SELECT id FROM analysis) ORDER BY created_at",
        "DROP TABLE feedback",
        "ALTER TABLE feedback_new RENAME TO feedback",
    ],
]
INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_feedback_analysis_id ON feedback (analysis_id)",
    "CREATE INDEX IF NOT EXISTS ix_job_owner_id ON job (owner_id)",
    "CREATE INDEX IF NOT EXISTS ix_job_status ON job (status)",
    "CREATE INDEX IF NOT EXISTS ix_analysis_owner_id ON analysis (owner_id)",
    "CREATE INDEX IF NOT EXISTS ix_analysis_owner_bowler_created ON analysis (owner_id, bowler_name, created_at)",
]
SCHEMA_VERSION = 1 + len(MIGRATIONS)

_engine = None
_engine_lock = threading.Lock()


def _migrate(engine) -> None:
    with engine.begin() as c:
        existing = {r[0] for r in c.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")}
        version = c.exec_driver_sql("PRAGMA user_version").scalar() or 0
        if not existing:  # fresh database: create the current schema directly
            SQLModel.metadata.create_all(c)
            version = SCHEMA_VERSION
        elif version == 0:
            version = 1  # pre-migration database created by create_all
        c.exec_driver_sql("PRAGMA foreign_keys=OFF")
        for i in range(version - 1, len(MIGRATIONS)):
            for stmt in MIGRATIONS[i]:
                c.exec_driver_sql(stmt)
        for stmt in INDEXES:
            c.exec_driver_sql(stmt)
        c.exec_driver_sql(f"PRAGMA user_version={SCHEMA_VERSION}")
        c.exec_driver_sql("PRAGMA foreign_keys=ON")


def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                settings.storage_dir.mkdir(parents=True, exist_ok=True)
                eng = create_engine(f"sqlite:///{settings.db_path}", connect_args={"check_same_thread": False})

                @event.listens_for(eng, "connect")
                def _pragma(dbapi_conn, _):
                    cur = dbapi_conn.cursor()
                    cur.execute("PRAGMA journal_mode=WAL")
                    cur.execute("PRAGMA busy_timeout=5000")
                    cur.execute("PRAGMA foreign_keys=ON")
                    cur.close()

                _migrate(eng)
                _engine = eng
    return _engine


def session() -> Session:
    return Session(get_engine())


def db_ok() -> bool:
    try:
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
