"""The Cosmos 3 de-normalization quantiles are graded, in whatever spelling they arrive.

:func:`~strands_robots.policies.cosmos3.action_decode.denormalize_quantile` is
handed three arrays: the model's normalized ``action`` and the domain's ``q01`` /
``q99`` physical quantiles. Only ``action`` was converted; the quantiles were
dereferenced raw (``q01.shape[-1]``), which decided two things this module never
meant to decide.

* **The documented explicit-stats path did not work.** Three registered
  embodiments ship no bundled quantiles, and ``load_action_stats`` raises telling
  the caller to supply that domain's own via ``stats=``. The bundled files store
  ``q01``/``q99`` as JSON arrays, so what a caller holds is a ``list`` - and a
  list has no ``.shape``, so the same content ``load_action_stats`` converts on
  the bundled path raised ``AttributeError: 'list' object has no attribute
  'shape'``, naming neither parameter. The width comparison sits *after* that
  dereference, so a wrong-width list was reported as a missing attribute rather
  than as the mismatch that comparison exists to report.
* **A non-finite quantile was accepted.** Every output column is
  ``0.5 * (a + 1) * (q99 - q01) + q01``, so one ``nan`` component spreads across
  the chunk, and :func:`decode_pose_trajectory` composes it into every pose:
  pre-fix a single ``nan`` quantile returned a full-length all-``nan`` SE3
  trajectory as a *result*. The first surface to refuse it was
  ``MinkIKBridge.solve``, which names the ``target_pose`` the decode built, not
  the stat that poisoned it.

Both are one decision - which values are a physical range - so both quantiles go
through :func:`~strands_robots.utils.finite_vector_error`, the shared vector
domain ``MinkIKBridge.solve`` itself and the other policy providers use, and the
conversion happens after the verdict rather than instead of it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.cosmos3 import action_decode
from strands_robots.policies.cosmos3.action_decode import (
    decode_pose_trajectory,
    denormalize_quantile,
    load_action_stats,
)

_DOMAIN = "droid_lerobot"
# Resolved from the installed module, not from the working directory, so the
# spelling under test is read out of the file the library actually ships.
_STATS_FILE = Path(str(action_decode.__file__)).parent / "stats" / f"{_DOMAIN}_stats.json"


def _bundled_lists() -> tuple[list[float], list[float]]:
    """The bundled quantiles exactly as ``json.load`` yields them: two lists.

    Read from the shipped file rather than transcribed, so the spelling under
    test stays the one a caller of the documented explicit-stats path actually
    holds.
    """
    raw = json.loads(_STATS_FILE.read_text())
    return raw["q01"], raw["q99"]


def _action(width: int) -> np.ndarray:
    """A deterministic normalized action chunk of ``width`` columns."""
    return np.random.default_rng(0).uniform(-1.0, 1.0, size=(6, width)).astype(np.float32)


# Spellings of one domain's quantiles that all describe the same physical ranges.
# A list is what the bundled JSON file holds; an ndarray is what
# ``load_action_stats`` builds from it. They must decode identically, because the
# only difference between them is which of the two paths the caller took.
def _accepted_spellings() -> list[tuple[str, Any, Any]]:
    q01, q99 = _bundled_lists()
    arr01, arr99 = np.asarray(q01, dtype=np.float32), np.asarray(q99, dtype=np.float32)
    return [
        ("list (the bundled file's own layout)", q01, q99),
        ("tuple", tuple(q01), tuple(q99)),
        ("ndarray (what load_action_stats builds)", arr01, arr99),
    ]


@pytest.mark.parametrize("label", [s[0] for s in _accepted_spellings()])
def test_every_spelling_of_one_domains_quantiles_decodes_identically(label: str) -> None:
    """A domain's quantiles rescale an action the same way however they are written."""
    spelling = {s[0]: (s[1], s[2]) for s in _accepted_spellings()}[label]
    bundled = load_action_stats(_DOMAIN)
    action = _action(len(bundled["q01"]))

    out = denormalize_quantile(action, *spelling)
    reference = denormalize_quantile(action, bundled["q01"], bundled["q99"])

    assert np.array_equal(out, reference), f"{label} decoded differently from the loader's arrays"
    assert np.isfinite(out).all()


# A value that cannot be a per-column physical range, and the parameter whose
# name the refusal has to carry. Kept as tuples rather than a dict because
# several entries differ only in which parameter holds the bad value.
def _refused_spellings() -> list[tuple[str, Any, Any, str]]:
    q01, q99 = _bundled_lists()
    arr01, arr99 = np.asarray(q01, dtype=np.float32), np.asarray(q99, dtype=np.float32)
    nan01 = list(q01)
    nan01[0] = math.nan
    inf99 = list(q99)
    inf99[2] = math.inf
    return [
        ("nan component, as a list", nan01, q99, "q01"),
        ("nan component, as an ndarray", np.asarray(nan01, dtype=np.float32), arr99, "q01"),
        ("inf component, as a list", q01, inf99, "q99"),
        ("inf component, as an ndarray", arr01, np.asarray(inf99, dtype=np.float32), "q99"),
        ("a scalar instead of a vector", 0.0, 1.0, "q01"),
        ("a 0-d ndarray", np.float32(0.0), np.float32(1.0), "q01"),
        ("numeric strings", ["0.0"] * len(q01), q99, "q01"),
        ("booleans", [False] * len(q01), q99, "q01"),
        ("None", None, None, "q01"),
        ("a generator, whose length cannot be read", (x for x in q01), q99, "q01"),
    ]


@pytest.mark.parametrize(
    ("label", "q01", "q99", "named"),
    [(s[0], s[1], s[2], s[3]) for s in _refused_spellings()],
)
def test_a_value_that_cannot_be_a_physical_range_is_refused_by_name(label: str, q01: Any, q99: Any, named: str) -> None:
    """The refusal names the quantile parameter and the surface, not an attribute.

    Pre-fix every one of these escaped as an ``AttributeError`` (or an
    ``IndexError`` raised *inside* the width comparison, for a 0-d array),
    naming neither ``q01`` nor ``q99``.
    """
    with pytest.raises(ValueError) as excinfo:
        denormalize_quantile(_action(10), q01, q99)

    message = str(excinfo.value)
    assert named in message, f"{label}: refusal does not name {named}: {message}"
    assert "denormalize_quantile" in message, f"{label}: refusal does not name the surface: {message}"


def test_the_same_bad_component_is_refused_the_same_way_in_either_quantile() -> None:
    """Neither quantile is graded more leniently than the other.

    They are read on one line and either one reaches every output column, so a
    domain that held only ``q01`` to the check would leave half the defect in
    place.
    """
    q01, q99 = _bundled_lists()
    poisoned01 = list(q01)
    poisoned01[4] = math.nan
    poisoned99 = list(q99)
    poisoned99[4] = math.nan

    with pytest.raises(ValueError) as low:
        denormalize_quantile(_action(len(q01)), poisoned01, q99)
    with pytest.raises(ValueError) as high:
        denormalize_quantile(_action(len(q01)), q01, poisoned99)

    assert str(low.value).replace("q01", "Q") != str(high.value)
    assert str(low.value).split("'q01'")[0] == str(high.value).split("'q99'")[0]


def test_a_poisoned_quantile_never_reaches_the_pose_trajectory() -> None:
    """No accepted decode can hand a non-finite pose to the IK bridge.

    The pre-fix behaviour this pins out: one ``nan`` quantile de-normalized
    without complaint and :func:`decode_pose_trajectory` composed it into all
    seven poses of a six-step chunk, returning an all-``nan`` trajectory as a
    successful result.
    """
    q01, q99 = _bundled_lists()
    poisoned = list(q01)
    poisoned[0] = math.nan
    action = _action(len(q01))

    with pytest.raises(ValueError):
        denormalize_quantile(action, poisoned, q99)

    clean = denormalize_quantile(action, q01, q99)
    trajectory = decode_pose_trajectory(clean[:, :9], np.eye(4, dtype=np.float32))
    assert trajectory.shape == (action.shape[0] + 1, 4, 4)
    assert np.isfinite(trajectory).all()


def test_a_wrong_width_list_reports_the_width_mismatch() -> None:
    """The width comparison is reachable for a quantile vector that is not an ndarray.

    Its whole job is to say that the stats do not describe every action column;
    pre-fix a list got as far as ``AttributeError: 'list' object has no
    attribute 'shape'`` instead.
    """
    q01, q99 = _bundled_lists()

    with pytest.raises(ValueError, match=r"does not match stats width"):
        denormalize_quantile(_action(len(q01)), q01[:4], q99[:4])
