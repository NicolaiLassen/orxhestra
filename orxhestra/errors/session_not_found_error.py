"""SessionNotFoundError - raised when a session cannot be located."""

from __future__ import annotations

from orxhestra.errors.not_found_error import NotFoundError


class SessionNotFoundError(NotFoundError):
    """Raised when a session ID does not exist in the session store.

    Parameters
    ----------
    session_id : str, optional
        The session ID that could not be found.

    See Also
    --------
    NotFoundError : Parent error class.
    BaseSessionService.get_session : Producer of this error when a
        session lookup fails.
    """

    def __init__(self, session_id: str = "") -> None:
        msg = f"Session '{session_id}' not found." if session_id else "Session not found."
        super().__init__(msg)
