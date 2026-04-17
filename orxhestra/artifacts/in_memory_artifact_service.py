"""In-memory artifact service — ephemeral storage for development.

Stores artifacts in a dictionary keyed by path.  Each path holds a
list of versioned entries.  Not thread-safe — for development and
testing only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from orxhestra.artifacts.base_artifact_service import (
    ArtifactVersion,
    BaseArtifactService,
)


@dataclass
class _ArtifactEntry:
    """A single versioned artifact entry."""

    data: bytes
    version_info: ArtifactVersion


class InMemoryArtifactService(BaseArtifactService):
    """In-memory artifact storage for development and testing.

    Artifacts are stored in a plain dictionary and lost when the
    process exits. Not suitable for production.

    See Also
    --------
    BaseArtifactService : Interface this implements.
    FileArtifactService : Persistent local-disk alternative.
    ArtifactVersion : Metadata carried on each stored version.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[_ArtifactEntry]] = {}

    def _path(
        self,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: str | None = None,
    ) -> str:
        if session_id:
            return f"{app_name}/{user_id}/{session_id}/{filename}"
        return f"{app_name}/{user_id}/{filename}"

    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        data: bytes,
        mime_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> int:
        """Save an artifact to memory.

        Returns
        -------
        int
            The new version number.
        """
        path = self._path(app_name, user_id, filename, session_id)
        versions = self._store.setdefault(path, [])
        version = len(versions)

        entry = _ArtifactEntry(
            data=data,
            version_info=ArtifactVersion(
                version=version,
                filename=filename,
                mime_type=mime_type,
                size_bytes=len(data),
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata=metadata or {},
            ),
        )
        versions.append(entry)
        return version

    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        version: int | None = None,
        session_id: str | None = None,
    ) -> bytes | None:
        """Load an artifact from memory.

        Returns
        -------
        bytes or None
            The artifact content, or ``None`` if not found.
        """
        path = self._path(app_name, user_id, filename, session_id)
        versions = self._store.get(path)
        if not versions:
            return None

        idx = version if version is not None else -1
        try:
            return versions[idx].data
        except IndexError:
            return None

    async def delete_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: str | None = None,
    ) -> bool:
        """Delete an artifact and all its versions.

        Returns
        -------
        bool
            ``True`` if the artifact existed and was deleted.
        """
        path = self._path(app_name, user_id, filename, session_id)
        if path in self._store:
            del self._store[path]
            return True
        return False

    async def list_artifact_keys(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str | None = None,
    ) -> list[str]:
        """List all artifact filenames.

        When ``session_id`` is provided, returns both session-scoped
        and user-scoped artifacts (merged, deduplicated).

        Returns
        -------
        list[str]
            Filenames of saved artifacts.
        """
        user_prefix: str = f"{app_name}/{user_id}/"
        filenames: set[str] = set()

        for path in self._store:
            if not path.startswith(user_prefix):
                continue
            remainder: str = path[len(user_prefix):]

            if "/" not in remainder:
                # User-scoped artifact
                filenames.add(remainder)
            elif session_id and remainder.startswith(f"{session_id}/"):
                # Session-scoped artifact matching requested session
                name: str = remainder[len(session_id) + 1:]
                if "/" not in name:
                    filenames.add(name)

        return sorted(filenames)

    async def list_versions(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: str | None = None,
    ) -> list[int]:
        """List all version numbers for an artifact.

        Returns
        -------
        list[int]
            Available version numbers.
        """
        path = self._path(app_name, user_id, filename, session_id)
        versions = self._store.get(path)
        if not versions:
            return []
        return list(range(len(versions)))

    async def get_artifact_version(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        version: int | None = None,
        session_id: str | None = None,
    ) -> ArtifactVersion | None:
        """Get metadata for a specific artifact version.

        Returns
        -------
        ArtifactVersion or None
            Version metadata, or ``None`` if not found.
        """
        path = self._path(app_name, user_id, filename, session_id)
        versions = self._store.get(path)
        if not versions:
            return None

        idx = version if version is not None else -1
        try:
            return versions[idx].version_info
        except IndexError:
            return None
