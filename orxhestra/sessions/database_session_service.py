"""Database-backed session service using SQLAlchemy async.

Supports any async database backend (SQLite via aiosqlite, PostgreSQL
via asyncpg, etc.).  Sessions and events are persisted to the database,
giving production-ready checkpointing and resume.

Usage::

    from orxhestra.sessions import DatabaseSessionService

    # SQLite (zero-config, local file):
    service = DatabaseSessionService("sqlite+aiosqlite:///sessions.db")
    await service.initialize()

    # PostgreSQL:
    service = DatabaseSessionService(
        "postgresql+asyncpg://user:pass@localhost/mydb"
    )
    await service.initialize()
"""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from orxhestra.events.event import Event
from orxhestra.sessions.base_session_service import BaseSessionService
from orxhestra.sessions.session import Session


class DatabaseSessionService(BaseSessionService):
    """Persistent session service backed by a SQL database.

    Uses SQLAlchemy async engine under the hood.  Call ``initialize()``
    once at startup to create the schema.

    Parameters
    ----------
    connection_string : str
        SQLAlchemy async connection string, e.g.
        ``"sqlite+aiosqlite:///sessions.db"`` or
        ``"postgresql+asyncpg://user:pass@host/db"``.
    """

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        self._engine: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """Create the database engine and tables if they don't exist."""
        from sqlalchemy import Column, Float, String, Text
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.orm import DeclarativeBase

        class Base(DeclarativeBase):
            pass

        class SessionRow(Base):
            __tablename__ = "orx_sessions"
            id = Column(String, primary_key=True)
            app_name = Column(String, nullable=False, index=True)
            user_id = Column(String, nullable=False, index=True)
            state_json = Column(Text, default="{}")
            last_update_time = Column(Float, default=0.0)

        class EventRow(Base):
            __tablename__ = "orx_events"
            id = Column(String, primary_key=True)
            session_id = Column(String, nullable=False, index=True)
            event_json = Column(Text, nullable=False)
            created_at = Column(Float, nullable=False)

        self._engine = create_async_engine(self._connection_string)
        self._SessionRow = SessionRow
        self._EventRow = EventRow
        self._Base = Base

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._initialized = True

    def _check_init(self) -> None:
        """Raise if ``initialize()`` has not been called."""
        if not self._initialized:
            msg = "Call await service.initialize() before using DatabaseSessionService"
            raise RuntimeError(msg)

    def _make_session(self) -> Any:
        """Return a new SQLAlchemy async session factory."""
        from sqlalchemy.ext.asyncio import AsyncSession as SASession
        from sqlalchemy.orm import sessionmaker

        return sessionmaker(self._engine, class_=SASession, expire_on_commit=False)

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create a new session and persist it to the database."""
        self._check_init()

        sid = session_id or str(uuid4())
        session = Session(
            id=sid,
            app_name=app_name,
            user_id=user_id,
            state=state or {},
            last_update_time=time.time(),
        )

        row = self._SessionRow(
            id=sid,
            app_name=app_name,
            user_id=user_id,
            state_json=json.dumps(session.state),
            last_update_time=session.last_update_time,
        )

        async with self._make_session()() as db:
            db.add(row)
            await db.commit()

        return session

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> Session | None:
        """Retrieve a session and its events from the database."""
        self._check_init()
        from sqlalchemy import select

        async with self._make_session()() as db:
            result = await db.execute(
                select(self._SessionRow).where(
                    self._SessionRow.id == session_id,
                    self._SessionRow.app_name == app_name,
                    self._SessionRow.user_id == user_id,
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None

            events_result = await db.execute(
                select(self._EventRow)
                .where(self._EventRow.session_id == session_id)
                .order_by(self._EventRow.created_at)
            )
            event_rows = events_result.scalars().all()

        events = [Event.model_validate_json(er.event_json) for er in event_rows]

        return Session(
            id=row.id,
            app_name=row.app_name,
            user_id=row.user_id,
            state=json.loads(row.state_json),
            events=events,
            last_update_time=row.last_update_time,
        )

    async def append_event(self, session: Session, event: Event) -> Event:
        """Persist the event to the database and apply state delta."""
        event = await super().append_event(session, event)

        if event.partial:
            return event

        self._check_init()
        from sqlalchemy import update

        event_row = self._EventRow(
            id=event.id,
            session_id=session.id,
            event_json=event.model_dump_json(),
            created_at=event.timestamp,
        )

        async with self._make_session()() as db:
            db.add(event_row)
            await db.execute(
                update(self._SessionRow)
                .where(self._SessionRow.id == session.id)
                .values(
                    state_json=json.dumps(session.state),
                    last_update_time=session.last_update_time,
                )
            )
            await db.commit()

        return event

    async def update_session(
        self,
        session_id: str,
        *,
        state: dict[str, Any] | None = None,
    ) -> Session:
        """Merge state updates and persist to the database."""
        self._check_init()
        from sqlalchemy import select, update

        async with self._make_session()() as db:
            result = await db.execute(
                select(self._SessionRow).where(self._SessionRow.id == session_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                from orxhestra.errors import SessionNotFoundError

                raise SessionNotFoundError(session_id)

            current_state = json.loads(row.state_json)
            if state:
                current_state.update(state)

            now = time.time()
            await db.execute(
                update(self._SessionRow)
                .where(self._SessionRow.id == session_id)
                .values(
                    state_json=json.dumps(current_state),
                    last_update_time=now,
                )
            )
            await db.commit()

        return Session(
            id=row.id,
            app_name=row.app_name,
            user_id=row.user_id,
            state=current_state,
            last_update_time=now,
        )

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and all its events from the database."""
        self._check_init()
        from sqlalchemy import delete

        async with self._make_session()() as db:
            await db.execute(
                delete(self._EventRow).where(self._EventRow.session_id == session_id)
            )
            await db.execute(
                delete(self._SessionRow).where(self._SessionRow.id == session_id)
            )
            await db.commit()

    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: str,
    ) -> list[Session]:
        """List all sessions for a given app and user."""
        self._check_init()
        from sqlalchemy import select

        async with self._make_session()() as db:
            result = await db.execute(
                select(self._SessionRow).where(
                    self._SessionRow.app_name == app_name,
                    self._SessionRow.user_id == user_id,
                )
            )
            rows = result.scalars().all()

        return [
            Session(
                id=row.id,
                app_name=row.app_name,
                user_id=row.user_id,
                state=json.loads(row.state_json),
                last_update_time=row.last_update_time,
            )
            for row in rows
        ]
