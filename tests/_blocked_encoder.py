# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An install with no clip encoder, for one block, put back exactly as found.

Two registries answer for an optional module, so blocking one is not enough: the
``sys.modules`` entry is what makes ``import imageio`` fail, and
``strands_robots.utils``'s memo is what
:func:`~strands_robots.utils.require_optional` consults *before* importing at
all, so an earlier successful import in the same session answers from there
instead.

Both have to be **restored**, and restoring the entry means putting back the
object it displaced rather than deleting the key. A deleted key does not undo an
import - it orphans every reference already bound to that module, and the next
import runs the package again and returns a different object. Measured on the
ordering the suite collects in, with the entry deleted instead of restored:
:mod:`tests.simulation.test_policy_runner_video_writer_cleanup` patched
``get_writer`` on the module it imported at collection time while the rollout's
own ``require_optional("imageio")`` reached a fresh copy, so the spy was never
consulted, the real writer ran, and the cell that exists to pin "the writer is
closed when the rollout raises" reported a leak - a cell that passes alone and
fails behind its sibling.

Two callers block the encoder, in :mod:`tests.simulation.mujoco` and
:mod:`tests.simulation.isaac`, and they had a copy of this each. One owner is
what keeps the pair of registries - and the restoration - from being rediscovered
per copy. :mod:`tests.test_sys_modules_removal_leaves_no_orphan` grades the rule
for the whole test tree.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator

import pytest

from strands_robots import utils

#: The module :func:`strands_robots.rendering.video.encode_clip` requires.
ENCODER_MODULE = "imageio"


@contextlib.contextmanager
def blocked_encoder() -> Iterator[None]:
    """Make importing the clip encoder fail for the duration of the block.

    ``monkeypatch`` performs the restoration, so both registries are put back on
    the exception path too: a ``None`` entry is what makes the import raise, and
    the memo is dropped so an earlier import cannot answer in its place.

    Yields:
        Nothing. Inside the block ``require_optional(ENCODER_MODULE)`` raises
        ``ImportError`` exactly as it does on an install without the extra.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, ENCODER_MODULE, None)  # type: ignore[arg-type]
        patch.delitem(utils._lazy_modules, ENCODER_MODULE, raising=False)
        yield
