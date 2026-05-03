"""Property-based test for ``_sanitize_name`` XML-injection safety.

``_sanitize_name`` is the single gate between user-supplied names and
MJCF XML attribute values. A regex escape would allow XML injection at
any of the 5 interpolation sites in ``mjcf_builder.py`` and
``scene_ops.py``. This test fuzzes the function with ~5000 random
inputs and asserts one of two outcomes:

  1. ``ValueError`` raised (name rejected), OR
  2. output is ``==`` to the input AND contains no XML-dangerous chars.

If the regex ever loosens to allow ``<``, ``>``, ``"``, ``'``, ``&``
through, or if output drifts from input (which would break name lookup
downstream), this test fails.

No ``hypothesis`` dep — hand-rolled brute-force is enough for a single
regex contract and avoids pulling a fuzzing library into dev deps.
"""

from __future__ import annotations

import random
import string

import pytest

from strands_robots.simulation.mujoco.mjcf_builder import _sanitize_name

# Characters an MJCF XML attribute value MUST NOT contain verbatim.
_XML_DANGEROUS = set("<>&\"'")

# Every printable ASCII char — the full universe the regex must classify.
_PRINTABLE = string.printable


def _random_name(rng: random.Random, max_len: int = 140) -> str:
    """Generate a random string up to ``max_len`` chars from printable ASCII.

    Intentionally biased to include long strings and edge chars
    (quotes, angle brackets, ampersands, whitespace, control chars).
    """
    length = rng.randint(0, max_len)
    return "".join(rng.choice(_PRINTABLE) for _ in range(length))


class TestSanitizeNameXmlInjection:
    """Fuzz ``_sanitize_name`` — for any input it MUST either raise or return
    a value free of XML-dangerous chars, equal to the input."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 1337, 2026])
    def test_fuzz_never_lets_dangerous_chars_through(self, seed: int) -> None:
        """5 seeds × 1000 samples = 5000 fuzz iterations per test run."""
        rng = random.Random(seed)
        for _ in range(1000):
            name = _random_name(rng)
            try:
                out = _sanitize_name(name)
            except ValueError:
                continue  # acceptable — name rejected
            # Must be identity (downstream code looks names up by exact string).
            assert out == name, f"_sanitize_name returned {out!r} != input {name!r}; downstream lookup would break."
            # And the output MUST be XML-safe.
            bad = _XML_DANGEROUS.intersection(out)
            assert not bad, f"_sanitize_name accepted dangerous char(s) {bad!r} in {name!r}"

    @pytest.mark.parametrize(
        "payload",
        [
            # Classic XML-injection payloads against the 5 interpolation sites.
            'cube"><geom name="evil',
            "cube'/><body name='evil'/>",
            "cube<script>alert(1)</script>",
            "cube&amp;",
            "cube onmouseover=alert(1)",
            'name="x" rgba="1 0 0 1',
            "\x00",
            "\n<inject/>",
            "a" * 200,  # length-bomb
            "",  # empty
            " leading_space",
            "trailing_space ",
            ".starts_with_dot",
            "-starts_with_dash",
        ],
    )
    def test_rejects_known_injection_payloads(self, payload: str) -> None:
        """Every known-hostile payload must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid simulation name"):
            _sanitize_name(payload)

    @pytest.mark.parametrize(
        "name",
        [
            "cube",
            "robot_0",
            "arm0/shoulder_pan",
            "cam.front",
            "object-1",
            "_underscore_start",
            "a",
            "A1_2.3-4",
            "x" * 128,  # max length
        ],
    )
    def test_accepts_legitimate_names(self, name: str) -> None:
        """Names following the documented grammar round-trip unchanged."""
        # Grammar: ^[a-zA-Z0-9_][a-zA-Z0-9_.\-]{0,127}$
        # The `/` in `arm0/shoulder_pan` is NOT in the grammar — but MuJoCo
        # uses it as a namespace separator. Filter those out of this test.
        if "/" in name:
            with pytest.raises(ValueError):
                _sanitize_name(name)
            return
        assert _sanitize_name(name) == name

    def test_namespace_separator_is_rejected(self) -> None:
        """``/`` is a MuJoCo namespace separator — users must not pass it raw.

        Regression guard: the grammar intentionally excludes ``/`` so that
        ``arm0/shoulder_pan`` (which exists only in the injected XML, not
        in user input) can never originate from a user-controlled name.
        """
        with pytest.raises(ValueError):
            _sanitize_name("arm0/shoulder_pan")
