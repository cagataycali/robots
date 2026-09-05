"""The ``ready`` handshake's metadata is held to the domain a local policy is.

:class:`~strands_robots.inference.client.RemotePolicy` answers every
introspection probe with a number the peer sent it, so the handshake is where a
locally-loaded checkpoint's constructor sits in the remote arrangement -- and
:func:`~strands_robots.policies.base.chunk_count_error` exists precisely so
"the same chunk count cannot be refused by a local checkpoint and accepted by
the server serving it".

The two chunk counts used to arrive through a bare ``int()`` and the two
capability flags through a bare ``bool()``, which is silent rather than lenient:

* an advertised ``0`` landed behind :attr:`Policy.execution_horizon`'s
  ``max(1, int(...))`` floor, so a peer declaring a 16-action chunk was mirrored
  as single-step and :meth:`Policy.is_chunk_emitting` answered ``False`` -- the
  outcome ``chunk_count_error`` documents that floor as "silently destructive"
  for;
* an advertised ``8.9`` was truncated to ``8`` and a ``"16"`` parsed, neither of
  which a local checkpoint may pass;
* ``bool("no")`` is ``True``, so a peer answering ``"no"`` turned a capability
  ON;
* and ``null`` reached CPython, raising ``TypeError: int() argument must be a
  string...`` out of the middle of a connect, naming neither the field nor the
  peer.

``required_bodies`` arrived through a filter, which is quieter still: the mirror
KEPT the entries it could use and dropped the rest, so a peer advertising
``["torso_link", 42]`` produced a proxy declaring ``("torso_link",)`` -- a
declaration nobody made. Nothing downstream can tell that from a peer that
really asked for one body: the robot host resolves the shorter set against its
scene, merges poses for it, and reports a successful rollout, while the served
tracker's other anchor link never arrives and it reads ``base_quat`` -- the
pelvis -- in its place. The local owner
:func:`~strands_robots.policies.base.collect_required_bodies` refuses that same
list by name, and its docstring is explicit that the two surfaces "must not
disagree".

The sibling module ``test_remote_policy_handshake_contract.py`` pins the loud
seams of a misbehaving peer; these pin the quiet ones.
"""

from typing import Any

import pytest

from strands_robots.inference import RemotePolicy, protocol
from strands_robots.policies.base import chunk_count_error, collect_required_bodies
from strands_robots.policies.mock import MockPolicy

#: Every mirrored field, with the value a compliant peer sends.
_COMPLIANT: dict[str, Any] = {
    "provider_name": "lerobot_local",
    "requires_images": True,
    "actions_per_step": 16,
    "supports_rtc": False,
    "execution_horizon": 16,
    "required_bodies": ["torso_link", "pelvis"],
}

#: ``(advertised required_bodies, the fragment a refusal must quote, what the
#: filter mirrored instead)``. The last column is measured on the pre-fix client.
#: The first two rows are the ones the filter got WRONG rather than merely quiet:
#: it answered with a shorter declaration instead of none at all.
_UNMIRRORABLE_BODIES: list[tuple[Any, str, str]] = [
    (["torso_link", 42], "42", "mirrored ('torso_link',): a declaration the peer never sent"),
    (["torso_link", "  "], "'  '", "mirrored ('torso_link',): the blank name dropped"),
    (["torso_link", None], "None", "mirrored ('torso_link',): the null name dropped"),
    ("torso_link", "'torso_link'", "mirrored (): a bare str is not a list, so nothing applied"),
    (42, "42", "mirrored (): not a list, so nothing applied"),
    ({"torso_link": 1}, "{'torso_link': 1}", "mirrored (): a JSON object, so nothing applied"),
    ([42, None], "42", "mirrored (): every entry unusable"),
]


class _Declaring(MockPolicy):
    """A local policy stating ``required_bodies`` exactly as handed in."""

    def __init__(self, declared: Any) -> None:
        super().__init__()
        self._declared = declared

    @property
    def required_bodies(self) -> Any:
        return self._declared


#: ``(field, advertised value, what reading it unchecked produced)``. The last
#: column is measured on the pre-fix client and is what the refusal replaces.
_UNMIRRORABLE: list[tuple[str, Any, str]] = [
    ("execution_horizon", 0, "floored to 1: a 16-action chunk mirrored as single-step"),
    ("execution_horizon", -5, "floored to 1"),
    ("execution_horizon", 8.9, "truncated to 8"),
    ("execution_horizon", "16", "parsed: a str no local checkpoint may pass"),
    ("execution_horizon", True, "a bool counted as 1"),
    ("execution_horizon", None, "TypeError from int(), naming no field"),
    ("actions_per_step", 0, "mirrored as 0"),
    ("actions_per_step", None, "TypeError from int(), naming no field"),
    ("actions_per_step", "many", "ValueError from int(), naming no field"),
    ("requires_images", "false", 'bool("false") is True: frames rendered anyway'),
    ("supports_rtc", "no", 'bool("no") is True: RTC turned on'),
    ("provider_name", 12345, "reported as the remote provider's name"),
]


class _FakeConnection:
    """Minimal stand-in for a ``websockets.sync`` connection over seeded frames."""

    def __init__(self, frames: list[str]) -> None:
        self._frames = list(frames)
        self.sent: list[str] = []
        self.closed = False

    def recv(self, timeout: float | None = None) -> str:  # noqa: ARG002 - parity with the real API
        if not self._frames:
            raise AssertionError("client called recv() more times than the test seeded frames")
        return self._frames.pop(0)

    def send(self, text: str) -> None:
        self.sent.append(text)

    def close(self) -> None:
        self.closed = True


def _ready(metadata: dict[str, Any]) -> str:
    """A well-formed ``ready`` handshake advertising *metadata*."""
    return protocol.dumps(
        {"type": protocol.MSG_READY, "protocol_version": protocol.PROTOCOL_VERSION, "metadata": metadata}
    )


def _client(monkeypatch: pytest.MonkeyPatch, frames: list[str]) -> tuple[RemotePolicy, _FakeConnection]:
    """A client whose next connect yields *frames*, plus the connection it gets."""
    fake = _FakeConnection(frames)
    monkeypatch.setattr("websockets.sync.client.connect", lambda *a, **k: fake)
    return RemotePolicy(endpoint="ws://127.0.0.1:65535"), fake


def _advertising(monkeypatch: pytest.MonkeyPatch, field: str, value: Any) -> tuple[RemotePolicy, _FakeConnection]:
    """A client handed compliant metadata with one field replaced by *value*."""
    return _client(monkeypatch, [_ready({**_COMPLIANT, field: value})])


@pytest.mark.parametrize(("field", "value", "unchecked"), _UNMIRRORABLE, ids=lambda v: str(v)[:24])
def test_a_field_this_client_cannot_mirror_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any, unchecked: str
) -> None:
    """A value outside the local domain is a named refusal, not a quiet mirror."""
    client, _ = _advertising(monkeypatch, field, value)

    with pytest.raises(ConnectionError, match=field) as raised:
        _ = client.execution_horizon

    assert repr(value) in str(raised.value), f"the refusal does not quote the value ({unchecked})"


def test_compliant_metadata_is_still_mirrored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The values a ``PolicyServer`` actually sends pass through unchanged."""
    client, _ = _client(monkeypatch, [_ready(_COMPLIANT)])

    assert client.execution_horizon == 16
    assert client.actions_per_step == 16
    assert client.requires_images is True
    assert client.supports_rtc is False
    assert client.remote_provider_name == "lerobot_local"
    assert client.required_bodies == ("torso_link", "pelvis")
    assert client.is_chunk_emitting() is True


def test_metadata_omitting_a_field_keeps_this_client_s_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A peer advertising a subset is usable: an absent field is not a refusal."""
    client, _ = _client(monkeypatch, [_ready({"provider_name": "recording", "requires_images": False})])

    assert client.requires_images is False
    assert client.execution_horizon == 1
    assert client.actions_per_step == 1
    assert client.required_bodies == ()


def test_a_refused_handshake_leaves_the_mirror_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is applied when one field is refused, so no half-mirrored state is served."""
    # ``provider_name`` and ``requires_images`` are read before the offending
    # count in the pre-fix order, so they are what a partial apply would leak.
    client, _ = _advertising(monkeypatch, "execution_horizon", 0)

    with pytest.raises(ConnectionError):
        _ = client.execution_horizon

    assert client.remote_provider_name == "unknown"
    assert client.actions_per_step == 1


def test_a_refused_handshake_does_not_leave_the_connection_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rejected connection is closed and dropped, not served on afterwards."""
    client, fake = _advertising(monkeypatch, "supports_rtc", "no")

    with pytest.raises(ConnectionError):
        _ = client.execution_horizon

    assert fake.closed is True
    assert client._ws is None


def test_a_refused_reset_re_advertisement_does_not_leave_the_connection_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server re-advertising an unmirrorable count on ``reset`` is refused the same way."""
    client, fake = _client(
        monkeypatch,
        [
            _ready(_COMPLIANT),
            protocol.dumps({"type": protocol.MSG_OK, "metadata": {**_COMPLIANT, "execution_horizon": 0}}),
        ],
    )
    assert client.execution_horizon == 16  # connected on compliant metadata

    with pytest.raises(ConnectionError, match="execution_horizon"):
        client.reset(seed=0)

    assert fake.closed is True
    assert client._ws is None


@pytest.mark.parametrize("value", [16, 1, 0, -5, 8.9, "16", True, None, [16]], ids=repr)
def test_the_wire_and_the_constructor_agree_about_a_chunk_count(monkeypatch: pytest.MonkeyPatch, value: Any) -> None:
    """One value, one verdict: what a checkpoint's constructor refuses, the wire refuses.

    Grounded in ``chunk_count_error`` itself -- the domain
    :class:`~strands_robots.policies.lerobot_local.policy.LerobotLocalPolicy`
    and ``LerobotAsyncPolicy`` hold their ``actions_per_step`` to -- rather than
    in a list of values this test picked, so the two cannot drift apart.
    """
    constructor_refuses = chunk_count_error(value, "actions_per_step", "lerobot_local") is not None
    client, _ = _advertising(monkeypatch, "actions_per_step", value)

    try:
        _ = client.execution_horizon
        wire_refuses = False
    except ConnectionError:
        wire_refuses = True

    assert wire_refuses is constructor_refuses


@pytest.mark.parametrize(("advertised", "quoted", "filtered"), _UNMIRRORABLE_BODIES, ids=lambda v: str(v)[:24])
def test_a_declaration_this_client_cannot_mirror_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, advertised: Any, quoted: str, filtered: str
) -> None:
    """A body name the runtime could not resolve is a named refusal, not a shorter list."""
    client, _ = _advertising(monkeypatch, "required_bodies", advertised)

    with pytest.raises(ConnectionError, match="required_bodies") as raised:
        _ = client.required_bodies

    assert quoted in str(raised.value), f"the refusal does not quote the offending value ({filtered})"


def test_a_refused_declaration_does_not_leave_the_connection_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rejected connection is closed and dropped, exactly as for a refused count."""
    client, fake = _advertising(monkeypatch, "required_bodies", ["torso_link", 42])

    with pytest.raises(ConnectionError):
        _ = client.required_bodies

    assert fake.closed is True
    assert client._ws is None
    assert client._required_bodies == (), "a refused handshake half-applied the declaration"


@pytest.mark.parametrize(
    "declared",
    [
        ["torso_link", "pelvis"],
        ["torso_link"],
        [],
        ["torso_link", "torso_link"],
        ["torso_link", 42],
        ["torso_link", "  "],
        ["torso_link", None],
        "torso_link",
        42,
        {"torso_link": 1},
        [42, None],
    ],
    ids=repr,
)
def test_the_wire_and_the_local_declaration_agree_about_a_body_list(
    monkeypatch: pytest.MonkeyPatch, declared: Any
) -> None:
    """One declaration, one verdict: what a local policy may not declare, the wire refuses.

    Graded against ``collect_required_bodies`` -- the owner the simulation
    runtime and ``PolicyServer`` both ask -- rather than against a list of values
    this test picked, so the two halves cannot drift apart. The accepting rows
    matter as much as the refusing ones: an empty list and a repeated name are
    both things a policy may declare in-process, so refusing them on the wire
    would be the same disagreement in the other direction.
    """
    try:
        collect_required_bodies(_Declaring(declared))
        local_refuses = False
    except TypeError:
        local_refuses = True

    client, _ = _advertising(monkeypatch, "required_bodies", declared)
    try:
        _ = client.required_bodies
        wire_refuses = False
    except ConnectionError:
        wire_refuses = True

    assert wire_refuses is local_refuses
