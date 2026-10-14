"""What a world needs from a body it holds."""

from __future__ import annotations

from typing import Protocol

from ..description import Description


class Member(Protocol):
    """What a world needs from a robot it holds: an id, a Description, a hook for when it is removed.

    ``Robot`` is the one implementation; the world asks for nothing more, so it never imports it.
    """

    id: str
    description: Description

    def detached(self) -> None: ...
