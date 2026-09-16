### Fixed: `Robot(..., mode="real")` without lerobot is refused with the `[lerobot]` extra

On a core-only install (`pip install strands-robots`) the README's real-arm
quickstart, `Robot("so101", mode="real", port=...)`, ended in the interpreter's
bare `ModuleNotFoundError: No module named 'lerobot'`: `HardwareRobot`'s first
lerobot import was a plain `from lerobot.robots.config import ...`, while every
other door a core install reaches - sim, dashboard, the Feetech bus - already
named its extra through `require_optional`. The import now goes through
`require_optional("lerobot", extra="lerobot", ...)`, so the refusal reads
`'lerobot' is required for real-mode robots built through the lerobot driver ...
pip install 'strands-robots[lerobot]'` and carries `ImportError.name ==
"lerobot"` (it was `"lerobot.robots"`). The purpose names the driver rather than
real mode at large, because `driver="strands"` builds a real robot through a
native driver and needs no lerobot at all.
