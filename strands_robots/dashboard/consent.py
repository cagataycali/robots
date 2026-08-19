"""Turn a safety refusal into something a human can answer.

The SDK refuses two classes of request on purpose:

* ``UntrustedRemoteCodeError`` — the provider would run code from a HuggingFace
  repo (``trust_remote_code=True``), so it demands ``STRANDS_TRUST_REMOTE_CODE=1``.
* the mesh command validator — ``pretrained_name_or_path`` is not covered by
  ``STRANDS_MESH_HF_REPO_ALLOW``.

Both are correct refusals and both arrive at the dashboard as *prose in an error
string*, which is a dead end for the person clicking the button: the only way
forward is a shell, an env var and a restart. This module parses that prose into
a structured consent request the API can attach to the failure, so the UI can
quote the risk and offer "Approve & retry" (U18).

Everything here is pure: no env mutation, no I/O. The caller decides whether the
human said yes, and :func:`env_patch` computes the *minimum* environment change
that grants exactly what was asked for — never a wildcard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

#: Same charset the mesh allowlist validator accepts for one entry
#: (``<org>`` or ``<org>/<repo>``). Duplicated deliberately: importing
#: ``strands_robots.mesh.security`` here would drag the mesh stack into the
#: dashboard's request path, and this module must stay import-cheap and pure.
_HF_ENTRY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$")

_TRUST_ENV = "STRANDS_TRUST_REMOTE_CODE"
_HF_ENV = "STRANDS_MESH_HF_REPO_ALLOW"
#: The teleop safety envelope: how far one frame may reach, and how fast a joint
#: may be commanded to travel. Both are RADIAN-shaped by default (4*pi / 8*pi),
#: so a degree-reporting arm (every SO-10x) has every frame refused.
_TELEOP_VALUE_ENV = "STRANDS_MESH_INPUT_VALUE_ABS"
_TELEOP_SLEW_ENV = "STRANDS_MESH_INPUT_SLEW_ABS"
#: Degrees plus a percent gripper: 400 covers a multi-turn wrist with headroom
#: and still refuses a runaway three orders of magnitude out. Not "unlimited" -
#: the envelope is the point, only its UNIT was wrong.
_TELEOP_DEGREE_VALUE = "400"
_TELEOP_DEGREE_SLEW = "800"

_PROVIDER_RE = re.compile(r"provider '([^']{1,120})'")
#: Read the value WHOLE — up to the closing quote, or to whitespace when bare —
#: and validate afterwards. A charset-limited capture would silently truncate
#: ``'org/repo;rm -rf /'`` to ``org/repo``, i.e. grant a repository the operator
#: never asked for, which is the opposite of what a consent dialog is for.
_REPO_QUOTED_RE = re.compile(r"pretrained_name_or_path=(['\"])(.{0,250}?)\1")
_REPO_BARE_RE = re.compile(r"pretrained_name_or_path=([^\s,]{1,250})")
_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: The refusal text can be a whole traceback; keep the evidence bounded.
_MAX_MESSAGE = 2000


@dataclass(frozen=True)
class ConsentRequest:
    """One thing the operator can approve, with the risk in the SDK's own words."""

    kind: str  # "trust_remote_code" | "hf_repo_allow"
    scope: str  # stable id to persist an approval against
    title: str
    risk: str
    env_var: str
    subject: str | None = None
    message: str = ""
    grants: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "scope": self.scope,
            "title": self.title,
            "risk": self.risk,
            "env_var": self.env_var,
            "subject": self.subject,
            "message": self.message,
            "grants": list(self.grants),
        }


#: The only things an operator can be asked to approve. A client posting an
#: approval names one of these and (at most) a subject — never an env var and
#: never a value, because "which variable does this grant touch" is a decision
#: this module owns.
KINDS: tuple[str, ...] = ("trust_remote_code", "hf_repo_allow", "teleop_degree_units")


def build_request(kind: str, subject: object = None, message: str = "") -> ConsentRequest | None:
    """Construct a consent request from ``kind`` + ``subject``, validating both.

    This is the single constructor: :func:`classify_refusal` parses a refusal
    into these two arguments, and the approval endpoint rebuilds the request the
    same way from what the browser sent. So a hostile client can ask for an
    allowlist entry, but it cannot ask for a *different variable* or smuggle a
    value past :data:`_HF_ENTRY_RE`.
    """
    if kind not in KINDS:
        return None
    text = message.strip()[:_MAX_MESSAGE] if isinstance(message, str) else ""
    name = subject.strip() if isinstance(subject, str) and subject.strip() else None

    if kind == "trust_remote_code":
        if name is not None and not _PROVIDER_NAME_RE.match(name):
            name = None
        shown = name or "this policy provider"
        return ConsentRequest(
            kind=kind,
            scope="trust_remote_code",
            title=f"Run model code from HuggingFace ({shown})?",
            risk=(
                f"{shown} loads the model with trust_remote_code=True: code stored in the "
                "model repository executes on this machine, with your files and your robots. "
                "Approve only for organisations you trust."
            ),
            env_var=_TRUST_ENV,
            subject=name,
            message=text,
            grants=("run repository code for every policy load from now on",),
        )

    if kind == "teleop_degree_units":
        # Subject is the arm that was refused, for the dialog's wording only: the
        # envelope is a MACHINE-wide setting, so the scope deliberately is not
        # per-peer. Granting it for "this arm" and quietly applying it to every
        # spawned child would be a lie about what was approved.
        shown = name if name and _PROVIDER_NAME_RE.match(name) else "this arm"
        return ConsentRequest(
            kind=kind,
            scope="teleop_degree_units",
            title="Set the teleop envelope to degrees?",
            risk=(
                f"The mesh refuses every teleop frame from {shown} because its safety envelope "
                f"assumes RADIANS (4·pi ≈ 12.57) and the arm reports DEGREES (a wrist at 170). "
                f"Approving widens the envelope to {_TELEOP_DEGREE_VALUE} units and the per-joint "
                f"speed bound to {_TELEOP_DEGREE_SLEW} units/s for every teleop stream on this "
                "machine — a wider envelope means a single frame may command a longer reach, so a "
                "faulty leader can ask for a bigger move before the bound stops it. It stays an "
                "envelope: a runaway three orders of magnitude out is still refused."
            ),
            env_var=_TELEOP_VALUE_ENV,
            subject=name,
            message=text,
            grants=(
                f"{_TELEOP_VALUE_ENV}={_TELEOP_DEGREE_VALUE} (how far one frame may reach)",
                f"{_TELEOP_SLEW_ENV}={_TELEOP_DEGREE_SLEW} (how fast one joint may be driven)",
            ),
        )

    if name is not None and not _HF_ENTRY_RE.match(name):
        name = None  # unparseable/hostile: ask, but grant nothing automatically
    shown = name or "the requested model"
    return ConsentRequest(
        kind=kind,
        scope=f"hf_repo_allow:{name}" if name else "hf_repo_allow",
        title=f"Allow the model {shown}?",
        risk=(
            f"{shown} is not in this machine's HuggingFace allowlist, so the mesh refused "
            "to load it. Approving adds exactly this repository — no other org, no wildcard."
        ),
        env_var=_HF_ENV,
        subject=name,
        message=text,
        grants=(f"load {name}" if name else "nothing yet — the repository name could not be read",),
    )


def classify_refusal(text: object) -> ConsentRequest | None:
    """Recognise a *continuable* refusal in ``text``, else ``None``.

    Detection keys off the env var the SDK itself names, because that string is
    the refusal's contract with the operator — the surrounding wording changes
    between versions, the variable does not.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    message = text.strip()[:_MAX_MESSAGE]

    if _TRUST_ENV in message:
        m = _PROVIDER_RE.search(message)
        return build_request("trust_remote_code", m.group(1) if m else None, message)

    # The teleop refusal names no env var (it is a per-frame rejection, logged by
    # the follower), so it is recognised by its own words. Both halves of the
    # envelope lead to the same grant: they are one unit decision, not two.
    if "input frame value for" in message and "out of range" in message:
        return build_request("teleop_degree_units", None, message)
    if "input frame slew for" in message and "out of range" in message:
        return build_request("teleop_degree_units", None, message)

    if _HF_ENV in message:
        quoted = _REPO_QUOTED_RE.search(message)
        bare = _REPO_BARE_RE.search(message)
        repo = quoted.group(2) if quoted else (bare.group(1) if bare else None)
        return build_request("hf_repo_allow", repo, message)

    return None


def env_patch(request: ConsentRequest, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The smallest env change that grants ``request``, given the current ``env``.

    Returns ``{}`` when there is nothing safe to grant (an unparseable repo) or
    when the grant is already in place — an empty patch is the caller's signal
    that approving would change nothing, which usually means the refusal came
    from a process started before the last approval.
    """
    env = env or {}
    if request.kind == "trust_remote_code":
        if str(env.get(_TRUST_ENV, "")).strip().lower() in ("1", "true", "yes"):
            return {}
        return {_TRUST_ENV: "1"}

    if request.kind == "teleop_degree_units":
        patch = {}
        if str(env.get(_TELEOP_VALUE_ENV, "")).strip() != _TELEOP_DEGREE_VALUE:
            patch[_TELEOP_VALUE_ENV] = _TELEOP_DEGREE_VALUE
        if str(env.get(_TELEOP_SLEW_ENV, "")).strip() != _TELEOP_DEGREE_SLEW:
            patch[_TELEOP_SLEW_ENV] = _TELEOP_DEGREE_SLEW
        return patch

    if request.kind == "hf_repo_allow":
        repo = request.subject
        if not repo or not _HF_ENTRY_RE.match(repo):
            return {}
        current = [e.strip() for e in str(env.get(_HF_ENV, "")).split(",") if e.strip()]
        org = repo.split("/", 1)[0]
        # An existing broader entry (the org, or the repo itself) already covers it.
        if repo in current or org in current:
            return {}
        merged = current + [repo]
        return {_HF_ENV: ",".join(merged)}

    return {}


def revoke_patch(request: ConsentRequest, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The env change that takes a grant BACK, given the current ``env``.

    A promise the UI makes ("you can revoke this") has to be executable, so
    revocation is computed by the same module that computes the grant — and it
    is narrow in the same way: revoking one repository leaves the rest of the
    allowlist exactly as it was. ``{}`` means there was nothing to take back.

    An empty string is how a variable is *cleared* here rather than deleted,
    because the .env file is the durable record: an absent line would let a
    stale value from a shell profile or a launchd plist win the next restart,
    which is a revocation that silently does not hold.
    """
    env = env or {}
    if request.kind == "trust_remote_code":
        if str(env.get(_TRUST_ENV, "")).strip().lower() not in ("1", "true", "yes"):
            return {}
        return {_TRUST_ENV: ""}

    if request.kind == "teleop_degree_units":
        # Back to the SDK defaults by CLEARING both, not by writing 12.566...:
        # a number frozen here would silently override a future SDK default.
        patch = {}
        if str(env.get(_TELEOP_VALUE_ENV, "")).strip():
            patch[_TELEOP_VALUE_ENV] = ""
        if str(env.get(_TELEOP_SLEW_ENV, "")).strip():
            patch[_TELEOP_SLEW_ENV] = ""
        return patch

    if request.kind == "hf_repo_allow":
        repo = request.subject
        if not repo:
            return {}
        current = [e.strip() for e in str(env.get(_HF_ENV, "")).split(",") if e.strip()]
        if repo not in current:
            # The org may still cover it; say nothing changed rather than
            # silently widening or narrowing something the caller did not name.
            return {}
        return {_HF_ENV: ",".join(e for e in current if e != repo)}

    return {}


def attach_consent(payload: dict, *sources: object) -> dict:
    """Add ``needs_consent`` to an error ``payload`` if any source is continuable.

    Used at the seams where a refusal becomes an HTTP response (spawn failure,
    task error), so a caller that never heard of consent still gets its old
    body plus one extra key.
    """
    for source in sources:
        request = classify_refusal(source)
        if request is not None:
            payload["needs_consent"] = request.as_dict()
            break
    return payload
