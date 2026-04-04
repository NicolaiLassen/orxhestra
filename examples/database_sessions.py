"""Database sessions — SQLite persistence with DatabaseSessionService.

Prerequisites: ``pip install sqlalchemy aiosqlite``

Run::

    python examples/database_sessions.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from orxhestra.sessions import DatabaseSessionService
from orxhestra.sessions.session import Session


async def main() -> None:
    """Show the full session lifecycle with SQLite persistence."""
    db_path: Path = Path(tempfile.mkdtemp()) / "sessions.db"
    conn_str: str = f"sqlite+aiosqlite:///{db_path}"

    service = DatabaseSessionService(conn_str)
    await service.initialize()
    print(f"Database initialized at {db_path}")

    # --- Create a session ---
    session: Session = await service.create_session(
        app_name="demo-app",
        user_id="user-42",
        state={"language": "python", "theme": "dark"},
    )
    print(f"\nCreated session: {session.id}")
    print(f"  state: {session.state}")

    # --- Retrieve it ---
    loaded: Session | None = await service.get_session(
        app_name="demo-app",
        user_id="user-42",
        session_id=session.id,
    )
    assert loaded is not None
    print(f"\nLoaded session:  {loaded.id}")
    print(f"  state: {loaded.state}")

    # --- Update state ---
    updated: Session = await service.update_session(
        session.id, state={"theme": "light"},
    )
    print(f"\nUpdated session: {updated.id}")
    print(f"  state: {updated.state}")

    # --- Create a second session and list all ---
    await service.create_session(
        app_name="demo-app",
        user_id="user-42",
        state={"language": "rust"},
    )
    sessions: list[Session] = await service.list_sessions(
        app_name="demo-app", user_id="user-42",
    )
    print(f"\nAll sessions for user-42: {len(sessions)}")
    for s in sessions:
        print(f"  {s.id} — {s.state}")

    # --- Clean up ---
    await service.delete_session(session.id)
    remaining: list[Session] = await service.list_sessions(
        app_name="demo-app", user_id="user-42",
    )
    print(f"\nAfter delete: {len(remaining)} session(s) remaining")


if __name__ == "__main__":
    asyncio.run(main())
