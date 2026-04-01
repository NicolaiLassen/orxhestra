"""Base artifact service — abstract interface for file/blob storage.

Artifacts are versioned files that agents can save, load, and share
during execution.  They support two scoping levels:

- **User-scoped**: visible across all sessions for a user.
- **Session-scoped**: visible only within a specific session.

Each save creates a new version (0, 1, 2, …).  Loading without a
version number returns the latest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ArtifactVersion(BaseModel):
    """Metadata for a single artifact version.

    Attributes
    ----------
    version : int
        Zero-based version number.
    filename : str
        The artifact filename.
    mime_type : str
        MIME type of the artifact content.
    size_bytes : int
        Size of the artifact content in bytes.
    created_at : str
        ISO 8601 timestamp of when this version was created.
    metadata : dict[str, Any]
        User-defined metadata attached to the artifact.
    """

    version: int
    filename: str
    mime_type: str = "application/octet-stream"
    size_bytes: int = 0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseArtifactService(ABC):
    """Abstract interface for artifact storage.

    Subclasses implement the storage backend (in-memory, filesystem,
    cloud storage, etc.).  All methods accept ``app_name``, ``user_id``,
    and an optional ``session_id`` to scope artifacts.
    """

    @abstractmethod
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
        """Save an artifact and return its version number.

        Parameters
        ----------
        app_name : str
            Application identifier.
        user_id : str
            User identifier.
        filename : str
            Artifact filename (e.g. ``"report.md"``).
        data : bytes
            The artifact content.
        mime_type : str
            MIME type of the content.
        metadata : dict, optional
            User-defined metadata.
        session_id : str, optional
            If provided, the artifact is session-scoped.

        Returns
        -------
        int
            The version number of the saved artifact (0-based).
        """

    @abstractmethod
    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        version: int | None = None,
        session_id: str | None = None,
    ) -> bytes | None:
        """Load an artifact's content.

        Parameters
        ----------
        app_name : str
            Application identifier.
        user_id : str
            User identifier.
        filename : str
            Artifact filename.
        version : int, optional
            Specific version to load.  ``None`` loads the latest.
        session_id : str, optional
            If provided, looks in session scope first.

        Returns
        -------
        bytes or None
            The artifact content, or ``None`` if not found.
        """

    @abstractmethod
    async def delete_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: str | None = None,
    ) -> bool:
        """Delete an artifact and all its versions.

        Parameters
        ----------
        app_name : str
            Application identifier.
        user_id : str
            User identifier.
        filename : str
            Artifact filename.
        session_id : str, optional
            Scope to delete from.

        Returns
        -------
        bool
            ``True`` if the artifact existed and was deleted.
        """

    @abstractmethod
    async def list_artifact_keys(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str | None = None,
    ) -> list[str]:
        """List all artifact filenames in a scope.

        Parameters
        ----------
        app_name : str
            Application identifier.
        user_id : str
            User identifier.
        session_id : str, optional
            If provided, includes session-scoped artifacts.

        Returns
        -------
        list[str]
            Filenames of saved artifacts.
        """

    @abstractmethod
    async def list_versions(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: str | None = None,
    ) -> list[int]:
        """List all version numbers for an artifact.

        Parameters
        ----------
        app_name : str
            Application identifier.
        user_id : str
            User identifier.
        filename : str
            Artifact filename.
        session_id : str, optional
            Scope to search.

        Returns
        -------
        list[int]
            Available version numbers.
        """

    @abstractmethod
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

        Parameters
        ----------
        app_name : str
            Application identifier.
        user_id : str
            User identifier.
        filename : str
            Artifact filename.
        version : int, optional
            Specific version.  ``None`` returns the latest.
        session_id : str, optional
            Scope to search.

        Returns
        -------
        ArtifactVersion or None
            Version metadata, or ``None`` if not found.
        """
