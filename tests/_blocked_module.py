# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An absent optional dependency for one block, restoring what it displaced.

A cell that grades what happens when an optional dependency is missing has to
make :func:`~strands_robots.utils.require_optional` fail, and that takes two
steps rather than one. ``sys.modules[name] = None`` is what makes ``import
name`` raise, and dropping :data:`strands_robots.utils._lazy_modules`' entry is
what stops a successful import earlier in the same session from answering from
the memo instead.

Both steps displace a value, and both have to put back exactly what they found
-- which is the part that is easy to get wrong, because the obvious teardown for
"I set an entry" is ``del``. Deleting the key does not undo the assignment: it
orphans every reference already bound to that module. A test module that did
``import imageio`` at collection time keeps the original object, while the next
import executes the package again and returns a **different** one::

    monkeypatch.setattr(imageio, "get_writer", spy)   # patches the original
    require_optional("imageio").get_writer(...)       # reaches the fresh copy

so the double is installed somewhere nothing will look, and the cell passes
while grading nothing. Measured after the absent-``imageio`` block that
``del``\\ ed its key, in the ordering the full suite collects: the two
``sys.modules`` objects differ, and
``tests/simulation/test_policy_runner_video_writer_cleanup.py`` reported "video
writer was leaked when the rollout raised" against a runner that closes it --
a real leak assertion answered by a spy that was never consulted. That file
passes in isolation.

One block, used everywhere, is what keeps the pairing right: three files had
copied the two-step idiom and two of the copies restored only one of the two
mappings. :mod:`tests.test_sys_modules_removal_leaves_no_orphan` grades that a
removal puts back what it displaced, so a fourth hand-rolled copy is reported
rather than merely discouraged.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator

from strands_robots import utils

#: Distinguishes "the mapping had no entry" from "the entry was ``None``".
#: ``None`` is a real, meaningful value here -- it is exactly what blocks the
#: import -- so it cannot double as the absent marker.
_ABSENT = object()


@contextlib.contextmanager
def blocked(name: str) -> Iterator[None]:
    """Make ``import name`` raise ``ImportError`` for the duration of the block.

    Both mappings ``require_optional`` consults are restored on the way out,
    whether the block returns or raises: each is put back to the value it held,
    or left with no entry if it had none.

    Args:
        name: Dotted module path to make unimportable, e.g. ``"imageio"``.
            Blocking a module the interpreter has already imported is the usual
            case, and is what the restoration exists for.

    Yields:
        Nothing. Inside the block ``require_optional(name)`` raises
        ``ImportError`` and ``import name`` fails with "halted; None in
        sys.modules".
    """
    displaced = sys.modules.get(name, _ABSENT)
    memoised = utils._lazy_modules.pop(name, _ABSENT)
    sys.modules[name] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        if displaced is _ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = displaced  # type: ignore[assignment]
        # Unconditional on both sides: an entry that was absent must be left
        # absent, not merely un-overwritten.
        if memoised is _ABSENT:
            utils._lazy_modules.pop(name, None)
        else:
            utils._lazy_modules[name] = memoised
