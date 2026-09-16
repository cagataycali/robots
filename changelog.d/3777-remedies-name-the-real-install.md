### Fixed: two `ImportError` remedies a caller could follow to no effect

Constructing a `CuroboPolicy` without cuRobo put `pip install
'strands-robots[curobo]'` first in its refusal. That extra is empty on purpose
(cuRobo is not on PyPI; the package there by that name is a squatter), so the
command exits 0 and installs nothing. The refusal now carries the source
checkout recipe and says why the extra is not the answer. Loading a raw
ProtoMotions `.pt` motion without torch pointed at the `[kimodo]` extra - a
different policy's whole stack - and now names `pip install torch` alone. A new
dependency-audit cell grades every `require_optional(extra=...)` site against
the extra's actual closure, so a refusal cannot name an extra that installs
nothing.
