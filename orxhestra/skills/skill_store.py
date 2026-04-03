"""Skill store — abstract base class for skill storage backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from orxhestra.skills.skill import Skill

logger = logging.getLogger(__name__)


class BaseSkillStore(ABC):
    """Abstract base for skill storage backends."""

    @abstractmethod
    async def get_skill(self, skill_id: str) -> Skill | None:
        """Retrieve a skill by its unique ID.

        Parameters
        ----------
        skill_id : str
            The unique skill identifier.

        Returns
        -------
        Skill or None
            The matching skill, or None if not found.
        """
        ...

    @abstractmethod
    async def get_by_name(self, name: str) -> Skill | None:
        """Retrieve a skill by its name.

        Parameters
        ----------
        name : str
            The skill name to look up.

        Returns
        -------
        Skill or None
            The matching skill, or None if not found.
        """
        ...

    @abstractmethod
    async def list_skills(self) -> list[Skill]:
        """Return all available skills.

        Returns
        -------
        list[Skill]
            All registered skills.
        """
        ...

    async def get_skill_resource(self, skill_name: str, resource_path: str) -> str | None:
        """Read a resource file's content from a skill.

        Resolves the resource from the skill's ``base_path`` on disk.
        Guards against path traversal attacks.

        Parameters
        ----------
        skill_name : str
            The skill to read the resource from.
        resource_path : str
            Relative path within the skill directory.

        Returns
        -------
        str or None
            File content, or None if not found or inaccessible.
        """
        skill = await self.get_by_name(skill_name)
        if skill is None or skill.base_path is None:
            return None

        # Validate the resource is declared
        known_paths = {r.path for r in skill.resources}
        if resource_path not in known_paths:
            return None

        # Resolve and guard against path traversal
        base = skill.base_path.resolve()
        target = (skill.base_path / resource_path).resolve()
        if not target.is_relative_to(base):
            logger.warning(
                "Path traversal attempt blocked: %s in skill %s",
                resource_path,
                skill_name,
            )
            return None

        if not target.is_file():
            return None

        try:
            return target.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read resource %s: %s", target, exc)
            return None
