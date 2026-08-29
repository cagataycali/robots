"""The presign TTL a caller names by keyword is read like the one it names by env.

``CameraOffloader`` resolves one quantity -- the lifetime of the presigned GET
URL it hands out per camera frame -- from two places, and they disagreed. The
environment path runs ``int(raw)`` on a string and falls back to the default
when that raises, so ``STRANDS_MESH_CAMERA_PRESIGN_TTL`` of ``2.5``, ``nan`` or
``inf`` each resolve to ``DEFAULT_PRESIGN_TTL_SECONDS``. The ``presign_ttl=``
keyword had no such step: it went straight to two *comparisons* against the
floor and the ceiling, and a comparison is permeable to anything that compares
false against both bounds.

``nan`` is the value that matters, because the ceiling it walks through is a
security bound: the module's own comment says it exists "to prevent accidental
day- or week-long URLs". ``botocore`` interpolates ``ExpiresIn`` into the
signature without reading it, so the URL carried ``X-Amz-Expires=nan`` and the
``/ref`` message published beside it carried ``expires_at: nan``. AWS refuses a
signed URL whose expiry field is not a number, so the frame is unreadable *and*
the window was never bounded.

What is deliberately unchanged is the *range*: ``0`` is still the documented
keyword-versus-environment precedence sentinel, ``-99`` is still a call-site bug
that clamps to ``1`` with a warning, and ``7200`` still clamps to the ceiling
(#262). Only readability moved, which is why
``_whole_seconds_or_none`` is sign-agnostic.
"""

from __future__ import annotations

import logging
import math

import pytest

from strands_robots.mesh.iot.camera_offload import (
    DEFAULT_PRESIGN_TTL_SECONDS,
    MAX_PRESIGN_TTL_SECONDS,
    CameraOffloader,
    _whole_seconds_or_none,
)

#: Spellings no count of seconds can be read from. Each one survived both
#: clamps (or, for ``inf``, tripped the ceiling with a notice that raised
#: inside ``logging``) and reached the signed URL.
UNREADABLE = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="negative-inf"),
    pytest.param(2.5, id="fractional"),
    pytest.param(1800.5, id="fractional-mid-range"),
    pytest.param(3600.5, id="fractional-just-over-the-ceiling"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
]

#: Values the module already resolved correctly, and must keep resolving the
#: same way. The last three are the #262 clamp contract.
UNCHANGED = [
    pytest.param(60, 60, id="the-default-named-explicitly"),
    pytest.param(3600, MAX_PRESIGN_TTL_SECONDS, id="exactly-the-ceiling"),
    pytest.param(1, 1, id="exactly-the-floor"),
    pytest.param(7200, MAX_PRESIGN_TTL_SECONDS, id="above-the-ceiling-clamps"),
    pytest.param(0, 1, id="the-precedence-sentinel-clamps-to-the-floor"),
    pytest.param(-99, 1, id="a-negative-call-site-bug-clamps-to-the-floor"),
]


def _offloader(monkeypatch, **kwargs):
    """Build an offloader with no environment override in play."""
    monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
    monkeypatch.delenv("STRANDS_MESH_CAMERA_S3_BUCKET", raising=False)
    return CameraOffloader(bucket="frames", **kwargs)


class TestPremises:
    """Facts that hold before and after, and are why the keyword needed reading."""

    def test_a_comparison_against_both_bounds_cannot_refuse_nan(self):
        """``nan`` passes the floor and the ceiling, so no clamp can catch it.

        This is the whole mechanism. The two guards are ``<`` and ``>`` tests,
        and every comparison against ``nan`` is false, so a guard built only
        from comparisons has nothing to say about it. The other unreadable
        spellings *are* caught by one bound or the other -- which is why they
        were resolved to a value the caller never named rather than passed
        straight through, a quieter failure but a failure of the same guard.
        """
        nan = float("nan")
        assert not (nan > MAX_PRESIGN_TTL_SECONDS)
        assert not (nan < 1)

    def test_the_other_unreadable_spellings_are_caught_by_exactly_one_bound(self):
        """Recorded so the two halves of the defect stay distinguishable.

        ``nan`` walked through the security ceiling untouched; the rest were
        resolved to some other number. Both are the comparison-only guard
        failing, and only the first is a URL with no bounded lifetime.
        """
        caught_by_a_bound = {
            value
            for value in (float("inf"), float("-inf"), 2.5, 1800.5, 3600.5, True, False)
            if value > MAX_PRESIGN_TTL_SECONDS or value < 1
        }
        assert caught_by_a_bound == {float("inf"), float("-inf"), 3600.5, False}

    def test_botocore_interpolates_the_expiry_without_reading_it(self):
        """A non-numeric ``ExpiresIn`` reaches the signed URL verbatim.

        Grounds the severity against the real consumer rather than against a
        stand-in: ``boto3`` ships in the ``mesh-iot`` extra this module needs,
        so this runs wherever the module is usable.
        """
        boto3 = pytest.importorskip("boto3")
        from botocore.config import Config

        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="x" * 40,
            config=Config(signature_version="s3v4"),
        )
        url = client.generate_presigned_url("get_object", Params={"Bucket": "b", "Key": "k"}, ExpiresIn=float("nan"))
        assert "X-Amz-Expires=nan" in url, (
            "botocore is expected to sign whatever it is handed, which is why the "
            "TTL has to be readable before it gets there"
        )

    @pytest.mark.parametrize("value", UNREADABLE)
    def test_the_environment_path_already_refused_these(self, monkeypatch, value):
        """The same spellings, named by env, already resolved to the default.

        This is the asymmetry the keyword now matches: one quantity, two ways
        to name it, and only one of them was read.
        """
        monkeypatch.delenv("STRANDS_MESH_CAMERA_S3_BUCKET", raising=False)
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", str(value))
        assert CameraOffloader(bucket="frames").presign_ttl == DEFAULT_PRESIGN_TTL_SECONDS


class TestAKeywordTtlIsReadBeforeItIsClamped:
    """The regression: an unreadable keyword no longer reaches the consumers."""

    @pytest.mark.parametrize("value", UNREADABLE)
    def test_an_unreadable_keyword_resolves_to_the_default(self, monkeypatch, value):
        assert _offloader(monkeypatch, presign_ttl=value).presign_ttl == DEFAULT_PRESIGN_TTL_SECONDS

    @pytest.mark.parametrize("value", UNREADABLE)
    def test_the_stored_ttl_is_always_a_finite_whole_number(self, monkeypatch, value):
        """What both consumers require, asserted on the stored value.

        ``ExpiresIn`` is signed verbatim and ``expires_at`` is published as
        ``ts + presign_ttl``, so a stored value that is not a finite integer is
        a non-number in a signed URL and on the wire.
        """
        ttl = _offloader(monkeypatch, presign_ttl=value).presign_ttl
        assert isinstance(ttl, int) and not isinstance(ttl, bool)
        assert math.isfinite(ttl)

    @pytest.mark.parametrize("value", UNREADABLE)
    def test_both_ways_of_naming_the_ttl_now_agree(self, monkeypatch, value):
        """One quantity, two spellings, one answer."""
        by_keyword = _offloader(monkeypatch, presign_ttl=value).presign_ttl
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", str(value))
        by_environment = CameraOffloader(bucket="frames").presign_ttl
        assert by_keyword == by_environment

    def test_a_fractional_ttl_is_refused_rather_than_truncated(self, monkeypatch):
        """2.5 seconds must not become 2 seconds.

        Truncating would honour a TTL the caller never named, which is the
        defect ``mesh.security._coerce_int`` was given the same split to avoid.
        """
        ttl = _offloader(monkeypatch, presign_ttl=2.5).presign_ttl
        assert ttl != 2
        assert ttl == DEFAULT_PRESIGN_TTL_SECONDS

    def test_the_ceiling_clamp_notice_no_longer_raises_inside_logging(self, monkeypatch, caplog):
        """``inf`` used to reach a notice rendered with ``%d``.

        ``"%d" % inf`` raises, so ``logging`` emitted ``--- Logging error ---``
        where the clamp notice belonged and the operator learnt nothing.
        """
        with caplog.at_level(logging.WARNING):
            _offloader(monkeypatch, presign_ttl=float("inf"))
        assert any("not a whole number of seconds" in record.getMessage() for record in caplog.records)

    def test_the_refusal_names_the_value_and_the_default(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            _offloader(monkeypatch, presign_ttl=2.5)
        text = " ".join(record.getMessage() for record in caplog.records)
        assert "2.5" in text
        assert str(DEFAULT_PRESIGN_TTL_SECONDS) in text


class TestWhatIsUnchanged:
    """Every readable spelling, including the whole #262 clamp contract."""

    @pytest.mark.parametrize(("value", "expected"), UNCHANGED)
    def test_a_readable_keyword_resolves_exactly_as_before(self, monkeypatch, value, expected):
        assert _offloader(monkeypatch, presign_ttl=value).presign_ttl == expected

    def test_an_integral_float_is_accepted_not_refused(self, monkeypatch):
        """``3600.0`` is how ``json.dumps`` renders an integer held in a float.

        Nothing is lost in reading it, so refusing it would break a config
        round-trip for no gain -- the split the wire-count guard applies too.
        """
        assert _offloader(monkeypatch, presign_ttl=3600.0).presign_ttl == MAX_PRESIGN_TTL_SECONDS

    def test_no_keyword_still_reads_the_environment(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_S3_BUCKET", raising=False)
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "120")
        assert CameraOffloader(bucket="frames").presign_ttl == 120

    def test_neither_source_leaves_the_default_unreachable(self, monkeypatch):
        assert _offloader(monkeypatch).presign_ttl == DEFAULT_PRESIGN_TTL_SECONDS


class TestTheReadabilityHelper:
    """``_whole_seconds_or_none`` decides readability, never range."""

    @pytest.mark.parametrize("value", UNREADABLE)
    def test_an_unreadable_value_answers_none(self, value):
        assert _whole_seconds_or_none(value) is None

    @pytest.mark.parametrize("value", [0, 1, -99, 60, 3600, 7200, 3600.0, -1.0])
    def test_a_whole_number_is_returned_as_an_int_whatever_its_sign(self, value):
        """Sign-agnostic on purpose: the clamps below own the range."""
        result = _whole_seconds_or_none(value)
        assert result == int(value)
        assert isinstance(result, int) and not isinstance(result, bool)

    @pytest.mark.parametrize("value", ["60", None, [60], {"ttl": 60}, object()])
    def test_a_non_number_answers_none(self, value):
        assert _whole_seconds_or_none(value) is None

    def test_a_count_wider_than_a_float_is_still_read(self):
        """``int``-only, so no float round-trip can refuse a large whole number.

        The clamp then bounds it; readability and range stay separate.
        """
        assert _whole_seconds_or_none(10**400) == 10**400
