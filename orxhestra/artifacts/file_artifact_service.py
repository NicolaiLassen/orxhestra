"""File-based artifact service — persistent local filesystem storage.

Stores artifacts as files in a directory tree::

    {root}/
    └── {app_name}/
        └── {user_id}/
            ├── {filename}/
            │   ├── 0           (version 0 payload)
            │   ├── 0.meta.json (version 0 metadata)
            │   ├── 1
            │   └── 1.meta.json
            └── sessions/
                └── {session_id}/
                    └── {filename}/
                        └── ...
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orxhestra.artifacts.base_artifact_service import (
    ArtifactVersion,
    BaseArtifactService,
)


class FileArtifactService(BaseArtifactService):
    """Artifact storage backed by the local filesystem.

    Each artifact version is stored as a separate file with an
    accompanying JSON metadata sidecar.

    Parameters
    ----------
    root : str or Path
        Root directory for artifact storage.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _artifact_dir(
        self,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: str | None = None,
    ) -> Path:
        base = self._root / app_name / user_id
        if session_id:
            base = base / "sessions" / session_id
        return base / filename

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
        """Save an artifact to the filesystem.

        Returns
        -------
        int
            The new version number.
        """
        artifact_dir = self._artifact_dir(app_name, user_id, filename, session_id)

        def _write() -> int:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(
                int(p.stem) for p in artifact_dir.glob("[0-9]*")
                if p.stem.isdigit() and not p.name.endswith(".meta.json")
            )
            version = (existing[-1] + 1) if existing else 0

            payload_path = artifact_dir / str(version)
            payload_path.write_bytes(data)

            meta = ArtifactVersion(
                version=version,
                filename=filename,
                mime_type=mime_type,
                size_bytes=len(data),
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata=metadata or {},
            )
            meta_path = artifact_dir / f"{version}.meta.json"
            meta_path.write_text(
                json.dumps({
                    "version": meta.version,
                    "filename": meta.filename,
                    "mime_type": meta.mime_type,
                    "size_bytes": meta.size_bytes,
                    "created_at": meta.created_at,
                    "metadata": meta.metadata,
                }, indent=2),
                encoding="utf-8",
            )
            return version

        return await asyncio.to_thread(_write)

    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        version: int | None = None,
        session_id: str | None = None,
    ) -> bytes | None:
        """Load an artifact from the filesystem.

        Returns
        -------
        bytes or None
            The artifact content, or ``None`` if not found.
        """
        artifact_dir = self._artifact_dir(app_name, user_id, filename, session_id)

        def _read() -> bytes | None:
            if not artifact_dir.exists():
                return None
            if version is not None:
                payload = artifact_dir / str(version)
            else:
                existing = sorted(
                    int(p.stem) for p in artifact_dir.glob("[0-9]*")
                    if p.stem.isdigit() and not p.name.endswith(".meta.json")
                )
                if not existing:
                    return None
                payload = artifact_dir / str(existing[-1])
            if not payload.exists():
                return None
            return payload.read_bytes()

        return await asyncio.to_thread(_read)

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
        artifact_dir = self._artifact_dir(app_name, user_id, filename, session_id)

        def _delete() -> bool:
            if not artifact_dir.exists():
                return False
            import shutil
            shutil.rmtree(artifact_dir)
            return True

        return await asyncio.to_thread(_delete)

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
        user_base: Path = self._root / app_name / user_id

        def _list() -> list[str]:
            names: set[str] = set()

            # User-scoped artifacts
            if user_base.exists():
                for d in user_base.iterdir():
                    if d.is_dir() and d.name != "sessions":
                        names.add(d.name)

            # Session-scoped artifacts
            if session_id:
                session_dir: Path = user_base / "sessions" / session_id
                if session_dir.exists():
                    for d in session_dir.iterdir():
                        if d.is_dir():
                            names.add(d.name)

            return sorted(names)

        return await asyncio.to_thread(_list)

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
        artifact_dir = self._artifact_dir(app_name, user_id, filename, session_id)

        def _versions() -> list[int]:
            if not artifact_dir.exists():
                return []
            return sorted(
                int(p.stem) for p in artifact_dir.glob("[0-9]*")
                if p.stem.isdigit() and not p.name.endswith(".meta.json")
            )

        return await asyncio.to_thread(_versions)

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
        artifact_dir = self._artifact_dir(app_name, user_id, filename, session_id)

        def _meta() -> ArtifactVersion | None:
            if not artifact_dir.exists():
                return None
            if version is not None:
                meta_path = artifact_dir / f"{version}.meta.json"
            else:
                existing = sorted(
                    int(p.stem) for p in artifact_dir.glob("[0-9]*")
                    if p.stem.isdigit() and not p.name.endswith(".meta.json")
                )
                if not existing:
                    return None
                meta_path = artifact_dir / f"{existing[-1]}.meta.json"
            if not meta_path.exists():
                return None
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            return ArtifactVersion(**raw)

        return await asyncio.to_thread(_meta)
