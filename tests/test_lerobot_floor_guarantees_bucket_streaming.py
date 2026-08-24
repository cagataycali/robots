#!/usr/bin/env python3
"""The declared lerobot floor must actually guarantee bucket streaming.

``stream_dataset(repo_type="bucket")`` needs a ``StreamingLeRobotDataset`` that
accepts a ``repo_type`` parameter. lerobot 0.6.0's constructor has none; 0.6.1
added ``repo_type: Literal["dataset", "bucket"]``. While the ``[lerobot]`` extra
floored lerobot at ``>=0.6.0`` the flagship path was only *docs*-guaranteed: the
resolver happily installed a lerobot that could not serve it at all, and the
runtime guard told the caller "not supported by any released lerobot" - naming no
remedy, because when that text was written none existed.

These tests pin the guarantee end to end:

* every lerobot-bearing extra floors at the version that accepts ``repo_type``,
  and *excludes* 0.6.0 rather than merely admitting a newer release;
* the packaging floor and the version the runtime guard advertises are the same
  value, so the remedy the error message names cannot drift from what the
  resolver installs;
* **executed, not asserted**: a lerobot that satisfies the declared floor really
  does accept ``repo_type``. A purely structural pin would keep passing if the
  capability moved to another release, leaving the floor citing a version that
  no longer delivers it;
* the guard is retained (an environment can carry a pre-existing older lerobot)
  and now names the upgrade as the remedy.
"""

from __future__ import annotations

import importlib.metadata as md
import inspect
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from strands_robots import streaming_dataset as sd
from strands_robots.streaming_dataset import BUCKET_STREAMING_MIN_LEROBOT

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# Every extra that carries its own lerobot requirement. ``molmoact2`` inherits
# the floor through ``strands-robots[lerobot]`` but ALSO declares
# ``lerobot[molmoact2]>=...`` directly, so a bump that skips it would leave a
# resolvable install of the older lerobot through that extra.
_LEROBOT_BEARING_EXTRAS = ("lerobot", "lerobot-async", "molmoact2")


def _extras() -> dict[str, list[str]]:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]


def _lerobot_requirements() -> dict[str, Requirement]:
    """The ``lerobot`` :class:`Requirement` each lerobot-bearing extra declares."""
    extras = _extras()
    out: dict[str, Requirement] = {}
    for name in _LEROBOT_BEARING_EXTRAS:
        assert name in extras, f"pyproject has no [{name}] extra"
        for spec in extras[name]:
            req = Requirement(spec)
            if req.name == "lerobot":
                out[name] = req
                break
        else:  # pragma: no cover - a missing pin is the failure this reports
            raise AssertionError(f"[{name}] extra declares no lerobot requirement")
    return out


def _installed_lerobot_or_skip() -> Version:
    """The installed lerobot version, or skip when it cannot be determined.

    Shaped so neither arm depends on ``pytest.skip`` being understood as
    ``NoReturn``: the version string is bound on *both* branches of the
    ``try``, and the function has a single explicit exit that every path
    reaches. A reader - human or static - can therefore see that no local is
    read before it is bound and that nothing falls off the end returning
    ``None``, without having to prove the skip cannot return.
    """
    pytest.importorskip("lerobot", reason="lerobot is an optional extra")
    try:
        raw: str | None = md.version("lerobot")
    except md.PackageNotFoundError:  # pragma: no cover - importable but unmetadata'd
        raw = None
    if raw is None:  # pragma: no cover - importable but unmetadata'd
        pytest.skip("lerobot version metadata unresolvable")
        return Version("0")  # unreachable; terminates the branch for static analysis
    return Version(raw)


def _declared_lerobot_specifier() -> SpecifierSet:
    """The version range the ``[lerobot]`` extra permits the resolver to install."""
    return SpecifierSet(str(_lerobot_requirements()["lerobot"].specifier))


class TestEveryExtraFloorsAtTheBucketStreamingVersion:
    """The resolver, not the docs, must guarantee the flagship streaming path."""

    def test_every_lerobot_bearing_extra_floors_at_or_above_the_bucket_version(self) -> None:
        """The declared lower bound must be at least the capability version.

        Asserted as a bound rather than as membership of one version: a later
        raise for an unrelated reason still guarantees bucket streaming, and a
        membership assertion would fail on it.
        """
        for extra, req in _lerobot_requirements().items():
            lower = min(Version(s.version) for s in req.specifier if s.operator == ">=")
            assert lower >= Version(BUCKET_STREAMING_MIN_LEROBOT), (
                f"[{extra}] floors lerobot at {lower}, below the {BUCKET_STREAMING_MIN_LEROBOT} "
                f"that first accepts repo_type: {req.specifier}"
            )

    def test_every_lerobot_bearing_extra_excludes_0_6_0(self) -> None:
        """Admitting the floor is not enough - 0.6.0 must be unresolvable.

        0.6.0 satisfies ``>=0.6.0`` and cannot serve bucket streaming, so a
        floor that merely *admits* 0.6.1 leaves the guarantee unenforced.
        """
        for extra, req in _lerobot_requirements().items():
            assert Version("0.6.0") not in req.specifier, (
                f"[{extra}] still admits lerobot 0.6.0, whose StreamingLeRobotDataset "
                f"takes no repo_type, so bucket streaming is not resolver-guaranteed: {req.specifier}"
            )

    def test_the_upper_cap_still_follows_the_minor_convention(self) -> None:
        """Raising the floor must not drop the ``<0.7.0`` cap (<1.0 caps minor)."""
        for extra, req in _lerobot_requirements().items():
            assert Version("0.7.0") not in req.specifier, f"[{extra}] lost the <0.7.0 cap: {req.specifier}"


class TestThePackagingFloorAndTheRuntimeGuardAgree:
    """One capability version, two consumers - they must not contradict."""

    def test_installing_the_extra_satisfies_the_remedy_the_guard_advertises(self) -> None:
        """The guard tells the caller to install ``strands-robots[lerobot]``.

        That remedy is only honest if the extra's floor is at least the version
        the guard names, so following the message provably clears the guard.
        """
        req = _lerobot_requirements()["lerobot"]
        lower = min(Version(s.version) for s in req.specifier if s.operator == ">=")
        assert lower >= Version(BUCKET_STREAMING_MIN_LEROBOT), (
            f"the guard tells callers to install the [lerobot] extra to get lerobot >= "
            f"{BUCKET_STREAMING_MIN_LEROBOT}, but that extra floors at {lower}: following "
            f"the advertised remedy would not clear the guard"
        )


class TestTheFloorReallyDeliversTheCapability:
    """Executed against the installed lerobot, not merely asserted."""

    def test_a_lerobot_satisfying_the_declared_floor_accepts_repo_type(self) -> None:
        """Any lerobot the declared floor admits must accept ``repo_type``.

        This is the half a structural pin cannot cover: if the capability ever
        moved to a different release, the floor would still read plausibly while
        no longer delivering what it exists to guarantee.
        """
        installed = _installed_lerobot_or_skip()
        specifier = _declared_lerobot_specifier()
        if installed not in specifier:
            pytest.skip(f"installed lerobot {installed} is outside the declared floor {specifier}")

        streaming_cls = sd._get_streaming_cls()
        params = inspect.signature(streaming_cls).parameters
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        assert "repo_type" in params or accepts_var_kw, (
            f"lerobot {installed} satisfies the declared floor {specifier} but its "
            f"StreamingLeRobotDataset does not accept repo_type, so the floor does not "
            f"guarantee bucket streaming"
        )

    def test_open_does_not_refuse_bucket_on_a_conforming_lerobot(self) -> None:
        """The guard must not fire for a lerobot the declared floor admits."""
        installed = _installed_lerobot_or_skip()
        if installed not in _declared_lerobot_specifier():
            pytest.skip(f"installed lerobot {installed} is outside the declared floor")

        # A stand-in mirroring the real constructor's repo_type parameter keeps
        # this off the network while still exercising the production predicate.
        class _Conforming:
            def __init__(self, repo_id: str, repo_type: str = "dataset", **_kw: object) -> None:
                self.repo_id = repo_id
                self.repo_type = repo_type
                self.num_frames = self.num_episodes = self.fps = 0

            def __iter__(self):  # pragma: no cover - not iterated here
                yield {}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sd, "StreamingLeRobotDataset", _Conforming, raising=False)
            reader = sd.StreamingDatasetReader.open("org/ds", repo_type="bucket", validate_deltas=False)
        assert reader.dataset.repo_type == "bucket"

    def test_an_unresolvable_installed_version_skips_rather_than_failing(self) -> None:
        """Not knowing what is installed is not a defect in the library.

        A lerobot that imports without distribution metadata cannot be placed
        against the declared floor, so the executed checks skip instead of
        reporting a failure the library is not responsible for. That arm is
        unreachable in a normal environment - hence its ``pragma: no cover`` -
        which is exactly why it is pinned here: it is the one path
        :func:`_installed_lerobot_or_skip` takes that no other test in this
        module reaches, and the arm whose control flow was reshaped to keep the
        helper free of a possibly-unbound local and an implicit fall-through.
        """

        def _no_distribution_metadata(name: str) -> str:
            raise md.PackageNotFoundError(name)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(md, "version", _no_distribution_metadata)
            with pytest.raises(pytest.skip.Exception, match="metadata unresolvable"):
                _installed_lerobot_or_skip()


class TestTheGuardIsRetainedAndNamesTheUpgrade:
    """A raised floor does not remove a pre-existing older lerobot from an env."""

    @staticmethod
    def _narrow_open() -> str:
        class _Narrow:
            def __init__(self, repo_id: str) -> None:
                raise AssertionError("constructor must never be reached")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sd, "StreamingLeRobotDataset", _Narrow, raising=False)
            with pytest.raises(RuntimeError) as exc:
                sd.StreamingDatasetReader.open("org/ds", repo_type="bucket", validate_deltas=False)
        return str(exc.value)

    def test_bucket_is_still_refused_when_the_constructor_lacks_repo_type(self) -> None:
        """Flooring the extra must not delete the runtime guard: an environment
        with a pre-existing older lerobot is still reachable, and silently
        opening the versioned-dataset namespace would read a different storage
        system."""
        assert "repo_type='bucket'" in self._narrow_open()

    def test_the_refusal_names_the_version_to_upgrade_to(self) -> None:
        text = self._narrow_open()
        assert BUCKET_STREAMING_MIN_LEROBOT in text, (
            f"the refusal must name the version that serves bucket streaming: {text!r}"
        )

    def test_the_refusal_names_an_install_command(self) -> None:
        text = self._narrow_open()
        assert "strands-robots[lerobot]" in text, f"the refusal must name a followable remedy: {text!r}"

    def test_the_refusal_no_longer_claims_no_release_supports_it(self) -> None:
        """The old text said "not supported by any released lerobot" - true when
        written, false since 0.6.1 shipped, and it left the caller with nothing
        to do."""
        text = self._narrow_open()
        assert "any released lerobot" not in text, f"the refusal still claims no release supports repo_type: {text!r}"


class TestDocsCiteTheDeclaredFloor:
    """The pages that promise bucket streaming must name the enforced version."""

    @pytest.mark.parametrize(
        "relpath",
        [
            "docs/recording.md",
            "docs/examples/overview.md",
            "examples/notebooks/README.md",
        ],
    )
    def test_bucket_streaming_pages_cite_the_declared_floor(self, relpath: str) -> None:
        text = (_REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert BUCKET_STREAMING_MIN_LEROBOT in text, (
            f"{relpath} promises bucket streaming but does not cite the enforced "
            f"lerobot floor {BUCKET_STREAMING_MIN_LEROBOT}"
        )
