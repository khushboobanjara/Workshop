"""Document visibility. Pure functions: no database, no web framework.

A document carries `min_role`, the LEAST privileged role that may retrieve it. A user with a more privileged role may
retrieve it too. Anything that is not a known human role (for example the ingest-only 'service' principal) sees nothing:
deny by default.
"""

LEVELS = ("viewer", "reviewer", "admin")           # least -> most privileged
_RANK = {name: i for i, name in enumerate(LEVELS)}


def is_level(value) -> bool:
    return isinstance(value, str) and value in _RANK


def visible_levels(role: str) -> tuple[str, ...]:
    """The min_role values a user with `role` may retrieve ('viewer' -> ('viewer',); 'admin' -> all three; unknown -> none)."""
    if role not in _RANK:
        return ()
    return LEVELS[: _RANK[role] + 1]


def can_see(role: str, min_role: str) -> bool:
    return min_role in visible_levels(role)
