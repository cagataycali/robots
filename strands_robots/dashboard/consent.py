"""Turn a safety refusal into something a human can answer.

The SDK refuses two classes of request on purpose:

* ``UntrustedRemoteCodeError`` - the provider would run code from a HuggingFace
  repo (``trust_remote_code=True``), so it demands ``STRANDS_TRUST_REMOTE_CODE=1``.
* the mesh command validator - ``pretrained_name_or_path`` is not covered by
  ``STRANDS_MESH_HF_REPO_ALLOW``.

Both are correct refusals and both arrive at the dashboard as *prose in an error
string*, which is a dead end for the person clicking the button: the only way
forward is a shell, an env var and a restart. This module parses that prose into
a structured consent request the API can attach to the failure, so the UI can
quote the risk and offer "Approve & retry" (U18).

Everything here is pure: no env mutation, no I/O. The caller decides whether the
human said yes, and :func:`env_patch` computes the *minimum* environment change
that grants exactly what was asked for - never a wildcard.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Mapping

#: Same charset the mesh allowlist validator accepts for one entry
#: (``<org>`` or ``<org>/<repo>``). Duplicated deliberately: importing
#: ``strands_robots.mesh.security`` here would drag the mesh stack into the
#: dashboard's request path, and this module must stay import-cheap and pure.
_HF_ENTRY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$")

_AGENT_TARGET_RE = re.compile(r"starting a task on ([A-Za-z0-9][A-Za-z0-9._:-]{0,63})")
_TRUST_ENV = "STRANDS_TRUST_REMOTE_CODE"
_HF_ENV = "STRANDS_MESH_HF_REPO_ALLOW"
#: The teleop safety envelope: how far one frame may reach, and how fast a joint
#: may be commanded to travel. Both are RADIAN-shaped by default (4*pi / 8*pi),
#: so a degree-reporting arm (every SO-10x) has every frame refused.
# Q80: the agent's own permission to START physical motion (dashboard/agent_motion.py owns the gate;
# this module owns the grant, so it is revocable on the permissions screen like everything else).
_AGENT_MOTION_ENV = "STRANDS_DASH_AGENT_PHYSICAL_MOTION"
#: Q81: the one entry here that is a LOCK, not a grant - a task POST must carry the browser's
#: confirmation. Surfaced on the permissions screen because that is where an operator goes to ask
#: what is different about this machine, and because a lock nobody can find is a lock nobody uses.
_TASK_CONFIRM_ENV = "STRANDS_DASH_TASK_REQUIRES_CONFIRM"
_TELEOP_VALUE_ENV = "STRANDS_MESH_INPUT_VALUE_ABS"
_TELEOP_SLEW_ENV = "STRANDS_MESH_INPUT_SLEW_ABS"
#: Q119. The SDK refuses FIVE things by allowlist and only ONE of them reached this module.
#: Measured by calling security.validate_command for each: pretrained_name_or_path was classified,
#: while policy_provider, policy_type and policy_host ended in a dead end - the exact complaint
#: ("these validation errors should be something I can continue over the UI") that consent exists to
#: answer. provider and type SHARE this variable, and the SDK's own sentence says so:
#: "Set STRANDS_MESH_POLICY_TYPE_ALLOW to extend (provider and policy_type share one allowlist)".
_POLICY_TYPE_ENV = "STRANDS_MESH_POLICY_TYPE_ALLOW"
#: Q119 part two, closing the family at 5 of 5. policy_host AND server_address's host check share
#: this one variable (is_safe_server_address strips scheme/path/port and defers to
#: is_safe_policy_host), so they are one kind here too.
_POLICY_HOST_ENV = "STRANDS_MESH_POLICY_HOST_ALLOW"
#: The SDK's own charset for an ENTRY in that variable (security._POLICY_HOST_ENTRY_RE), copied
#: rather than imported so this module stays free of the mesh at import time. Hostnames, IP
#: literals and CIDR ranges - and nothing shell-shaped.
_POLICY_HOST_ENTRY_RE = re.compile(r"^[A-Za-z0-9.:/_\-]{1,253}$")
#: Degrees plus a percent gripper: 400 covers a multi-turn wrist with headroom
#: and still refuses a runaway three orders of magnitude out. Not "unlimited" -
#: the envelope is the point, only its UNIT was wrong.
_TELEOP_DEGREE_VALUE = "400"
_TELEOP_DEGREE_SLEW = "800"

_PROVIDER_RE = re.compile(r"provider '([^']{1,120})'")
#: Read the value WHOLE - up to the closing quote, or to whitespace when bare -
#: and validate afterwards. A charset-limited capture would silently truncate
#: ``'org/repo;rm -rf /'`` to ``org/repo``, i.e. grant a repository the operator
#: never asked for, which is the opposite of what a consent dialog is for.
_REPO_QUOTED_RE = re.compile(r"pretrained_name_or_path=(['\"])(.{0,250}?)\1")
_REPO_BARE_RE = re.compile(r"pretrained_name_or_path=([^\s,]{1,250})")
_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
#: Taken only from the SDK's fixed `policy_provider=` / `policy_type=` prefix, quoted or bare, so a
#: paraphrase wrapped around the refusal cannot become the subject (the _AGENT_TARGET_RE rule).
_POLICY_HOST_SUBJECT_RE = re.compile(
    r"(policy_host|server_address)=(?:'([^']{1,300})'|\"([^\"]{1,300})\"|([^\s,]{1,300}))"
)
_POLICY_SUBJECT_RE = re.compile(r"policy_(?:provider|type)=(?:'([^']{1,80})'|\"([^\"]{1,80})\"|([^\s,]{1,80}))")

#: The refusal text can be a whole traceback; keep the evidence bounded.
_MAX_MESSAGE = 2000


def _host_entry(raw: object, *, strip_url: bool = False) -> str | None:
    """The allowlist ENTRY that grants ``raw``, or None if nothing safe can be derived.

    ``strip_url`` distinguishes the two refusals that share this variable, and MEASUREMENT decided
    it, not symmetry: is_safe_server_address strips scheme/path/port before checking, so a
    ``server_address='http://gpu.lan:8000'`` is granted by the entry ``gpu.lan``. But
    ``policy_host`` is matched LITERALLY, so ``policy_host='gpu.lan:8000'`` needs the entry
    ``gpu.lan:8000`` - granting ``gpu.lan`` there produced an approval that left the command
    refused, which is the exact "approval that changes nothing" this function exists to prevent. I
    wrote the symmetric version first and the round-trip test caught it.

    The refusal quotes what the operator sent - ``server_address='http://gpu.lan:8000'`` - but the
    variable holds hosts, matched literally by :func:`security.is_safe_policy_host`. Appending the
    URL verbatim would produce an entry that the SDK's own charset check rejects and that could
    never match anything: an approval that changes nothing, which is worse than a refusal because
    the operator believes they are through. So the scheme, path and port are stripped here, exactly
    as is_safe_server_address strips them.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not strip_url:
        # Already an entry (a policy_host, or a subject the browser sent back): validate, never
        # rewrite - the approval endpoint rebuilds the request from this string.
        return s if s and _POLICY_HOST_ENTRY_RE.match(s) else None
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0]  # path (a CIDR entry never arrives via a refusal, so / cannot be kept)
    if s.startswith("["):  # bracketed IPv6, with or without a port
        if "]" not in s:
            return None
        s = s[1 : s.index("]")]
    elif s.count(":") == 1:  # host:port - an IPv6 literal has more, and keeps them
        s = s.split(":", 1)[0]
    s = s.strip()
    if not s or not _POLICY_HOST_ENTRY_RE.match(s):
        return None
    return s


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

    @property
    def grantable(self) -> bool:
        """Would approving this actually change the environment?

        Q120. The UI had to guess this, and it guessed with a kind list: canApprove required a
        subject for ``hf_repo_allow`` (where the grant IS the subject) and said yes to everything
        else that named an env var. Correct until Q119 added two more allowlist kinds - after which
        a ``policy_host_allow`` with an unreadable host offered an ENABLED Approve button that
        would have written nothing, i.e. a security dialog claiming to have helped. The server owns
        env_patch, so the server answers the question; the client stops maintaining a list.
        """
        return bool(env_patch(self, {}))

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
            # Computed against an EMPTY env deliberately: the question is "is there a grant here at
            # all", not "is it already in place on this machine" - which env_patch answers for the
            # live env at approval time, and which must not disable the button (a refusal from a
            # process started before the last approval still needs its explanation).
            "grantable": self.grantable,
        }


#: The only things an operator can be asked to approve. A client posting an
#: approval names one of these and (at most) a subject - never an env var and
#: never a value, because "which variable does this grant touch" is a decision
#: this module owns.
KINDS: tuple[str, ...] = (
    "trust_remote_code",
    "hf_repo_allow",
    "teleop_degree_units",
    "agent_physical_motion",
    "policy_type_allow",
    "policy_host_allow",
)


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

    if kind == "agent_physical_motion":
        # Subject is the peer that was refused, for the wording only: the grant is MACHINE-wide,
        # because the gate reads one env var. Saying "allow it for this arm" and then letting the
        # agent drive the other one would be a lie about what was approved - the same reasoning as
        # teleop_degree_units above.
        # Null an unparseable name rather than only softening the WORDING: `subject` is echoed in
        # as_dict() and is what the approval endpoint rebuilds the request from, so a hostile string
        # kept here would travel further than the sentence that hid it. (trust_remote_code's rule;
        # caught by test_a_hostile_subject_is_dropped_not_echoed.)
        if name is not None and not _PROVIDER_NAME_RE.match(name):
            name = None
        shown = name or "a real robot"
        return ConsentRequest(
            kind=kind,
            scope="agent_physical_motion",
            title="Let the agent start motion on real robots?",
            risk=(
                f"The fleet agent asked to run a task on {shown}, which is real hardware. Approving "
                "lets it start physical motion on ANY real robot on this mesh from now on - by itself, "
                "from a chat sentence or a voice command, with no confirmation step and without the "
                "check that the policy fits the robot that the play button performs. It cannot see your "
                "room. Stopping is never gated either way, so 'everyone stop' works regardless."
            ),
            env_var=_AGENT_MOTION_ENV,
            subject=name,
            message=text,
            grants=("start tasks on real robots unattended, until you revoke it",),
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
                "machine - a wider envelope means a single frame may command a longer reach, so a "
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

    if kind == "policy_host_allow":
        # Subject is normalised to an ENTRY, not kept as the operator typed it: see _host_entry.
        host = _host_entry(name)  # validate only: classify_refusal already derived the entry
        shown = host or "that address"
        return ConsentRequest(
            kind=kind,
            scope=f"policy_host_allow:{host}" if host else "policy_host_allow",
            title=f"Let policies run on {shown}?",
            risk=(
                f"The mesh only talks to policy servers on loopback by default, and {shown} is not "
                "in the allowlist. Approving sends your robots' camera frames and joint states to "
                "that host, and lets the actions it returns drive real hardware - so it is trusted "
                "with what the arms SEE and what they DO. Hostnames are matched literally with no "
                "DNS resolution, so this trusts whatever that name resolves to at the time; an IP "
                "literal keeps the boundary under your control."
            ),
            env_var=_POLICY_HOST_ENV,
            subject=host,
            message=text,
            grants=(
                f"reach the policy server at {host}" if host
                else "nothing yet - the host could not be read",
            ),
        )

    if kind == "policy_type_allow":
        # One variable, two things the SDK refuses with it. The grant is deliberately NARROW - this
        # name only - mirroring hf_repo_allow: an operator approving "lerobot_async" must not be
        # opening every policy type the SDK can construct.
        if name is not None and not _PROVIDER_NAME_RE.match(name):
            name = None  # unparseable/hostile: ask, but grant nothing automatically
        shown = name or "the requested policy"
        return ConsentRequest(
            kind=kind,
            scope=f"policy_type_allow:{name}" if name else "policy_type_allow",
            title=f"Allow the policy {shown}?",
            risk=(
                f"{shown} is not in this machine's policy allowlist, so the mesh refused to build "
                "it. A policy decides what the arms DO, and approving lets this one be constructed "
                "and run on real hardware from now on. Exactly this name is added - no wildcard, "
                "and no other provider."
            ),
            env_var=_POLICY_TYPE_ENV,
            subject=name,
            message=text,
            grants=(
                f"build and run the policy {name}" if name
                else "nothing yet - the policy name could not be read",
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
            "to load it. Approving adds exactly this repository - no other org, no wildcard."
        ),
        env_var=_HF_ENV,
        subject=name,
        message=text,
        grants=(f"load {name}" if name else "nothing yet - the repository name could not be read",),
    )


def classify_refusal(text: object) -> ConsentRequest | None:
    """Recognise a *continuable* refusal in ``text``, else ``None``.

    Detection keys off the env var the SDK itself names, because that string is
    the refusal's contract with the operator - the surrounding wording changes
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

    if _AGENT_MOTION_ENV in message:
        # The refusal is ours (agent_motion.py), and it names the peer in the same breath as the
        # words MOVE REAL HARDWARE; take the peer only from that fixed prefix so the model's own
        # paraphrase around it cannot become the subject.
        m = _AGENT_TARGET_RE.search(message)
        return build_request("agent_physical_motion", m.group(1) if m else None, message)

    if _POLICY_HOST_ENV in message:
        m = _POLICY_HOST_SUBJECT_RE.search(message)
        subject = next((g for g in (m.groups()[1:] if m else ()) if g), None)
        # Which field was refused decides how the entry is derived - they are matched differently.
        entry = _host_entry(subject, strip_url=bool(m and m.group(1) == "server_address"))
        return build_request("policy_host_allow", entry, message)

    if _POLICY_TYPE_ENV in message:
        m = _POLICY_SUBJECT_RE.search(message)
        subject = next((g for g in (m.groups() if m else ()) if g), None)
        return build_request("policy_type_allow", subject, message)

    if _HF_ENV in message:
        quoted = _REPO_QUOTED_RE.search(message)
        bare = _REPO_BARE_RE.search(message)
        repo = quoted.group(2) if quoted else (bare.group(1) if bare else None)
        return build_request("hf_repo_allow", repo, message)

    return None


def env_patch(request: ConsentRequest, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The smallest env change that grants ``request``, given the current ``env``.

    Returns ``{}`` when there is nothing safe to grant (an unparseable repo) or
    when the grant is already in place - an empty patch is the caller's signal
    that approving would change nothing, which usually means the refusal came
    from a process started before the last approval.
    """
    env = env or {}
    if request.kind == "trust_remote_code":
        if str(env.get(_TRUST_ENV, "")).strip().lower() in ("1", "true", "yes"):
            return {}
        return {_TRUST_ENV: "1"}

    if request.kind == "agent_physical_motion":
        if str(env.get(_AGENT_MOTION_ENV, "")).strip().lower() in ("1", "true", "yes", "on"):
            return {}
        return {_AGENT_MOTION_ENV: "1"}

    if request.kind == "teleop_degree_units":
        patch = {}
        if str(env.get(_TELEOP_VALUE_ENV, "")).strip() != _TELEOP_DEGREE_VALUE:
            patch[_TELEOP_VALUE_ENV] = _TELEOP_DEGREE_VALUE
        if str(env.get(_TELEOP_SLEW_ENV, "")).strip() != _TELEOP_DEGREE_SLEW:
            patch[_TELEOP_SLEW_ENV] = _TELEOP_DEGREE_SLEW
        return patch

    if request.kind == "policy_host_allow":
        host = _host_entry(request.subject)
        if not host:
            return {}
        current = [e.strip() for e in str(env.get(_POLICY_HOST_ENV, "")).split(",") if e.strip()]
        if host in current:
            return {}
        return {_POLICY_HOST_ENV: ",".join(current + [host])}

    if request.kind == "policy_type_allow":
        name = request.subject
        if not name or not _PROVIDER_NAME_RE.match(name):
            return {}
        current = [e.strip() for e in str(env.get(_POLICY_TYPE_ENV, "")).split(",") if e.strip()]
        # No org shortcut here: a policy name has no hierarchy, so nothing broader can cover it.
        if name in current:
            return {}
        return {_POLICY_TYPE_ENV: ",".join(current + [name])}

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


def granted_state(env: Mapping[str, str] | None = None) -> dict:
    """What this machine currently grants - every kind, in one place.

    GET /api/consent used to build this inline and covered only two of the three kinds, so the
    teleop envelope widening (the grant with actual physical reach: it raises how far a single
    teleop frame may command an arm) was invisible on the permissions screen and therefore could
    not be revoked there - while the consent dialog promised it could. A grant with no surface is
    a grant nobody can take back.

    ``teleop_degree_units`` reports what the environment ACTUALLY holds, not merely whether it
    equals the value this module would have written: an operator who set a wider bound by hand
    must see that, and a half-set pair (reach widened, speed bound untouched) is reported as
    granted rather than hidden, because the widened half is already in force.
    """
    env = os.environ if env is None else env
    allow = [e.strip() for e in str(env.get(_HF_ENV, "")).split(",") if e.strip()]
    value_abs = str(env.get(_TELEOP_VALUE_ENV, "")).strip()
    slew_abs = str(env.get(_TELEOP_SLEW_ENV, "")).strip()
    return {
        "kinds": list(KINDS),
        "trust_remote_code": str(env.get(_TRUST_ENV, "")).strip().lower() in ("1", "true", "yes"),
        # Reported as what the environment ACTUALLY holds: a hand-set "on" is in force and must be
        # visible, or the screen would deny a permission the agent is currently using.
        "agent_physical_motion": (
            str(env.get(_AGENT_MOTION_ENV, "")).strip().lower() in ("1", "true", "yes", "on")
        ),
        "hf_repo_allow": allow,
        # Shown for the same reason the teleop envelope had to be: a grant with no surface cannot
        # be revoked, while the dialog promises it can.
        "policy_type_allow": [
            e.strip() for e in str(env.get(_POLICY_TYPE_ENV, "")).split(",") if e.strip()
        ],
        # Loopback is allowed by the SDK's own default and is NOT listed: this key answers "what has
        # this machine been opened up to", and printing localhost as a grant would bury the one
        # entry that matters among defaults nobody approved.
        "policy_host_allow": [
            e.strip() for e in str(env.get(_POLICY_HOST_ENV, "")).split(",") if e.strip()
        ],
        "teleop_degree_units": {
            "granted": bool(value_abs or slew_abs),
            "value_abs": value_abs or None,
            "slew_abs": slew_abs or None,
            # True only when it is exactly the pair this module grants; a hand-tuned wider bound
            # must not be described to the operator as "the degrees preset".
            "is_degree_preset": value_abs == _TELEOP_DEGREE_VALUE and slew_abs == _TELEOP_DEGREE_SLEW,
        },
        # Reported next to the grants but NOT as one: every other key here loosens something, this
        # tightens it. Kept in the same payload because the operator's question is one question -
        # "what is different about this machine?" - and answering half of it from another endpoint is
        # how the teleop envelope stayed invisible (see this function's own history).
        "locks": {
            "task_requires_confirm": (
                str(env.get(_TASK_CONFIRM_ENV, "")).strip().lower() in ("1", "true", "yes", "on")
            ),
            "task_requires_confirm_env": _TASK_CONFIRM_ENV,
        },
    }


def revoke_patch(request: ConsentRequest, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The env change that takes a grant BACK, given the current ``env``.

    A promise the UI makes ("you can revoke this") has to be executable, so
    revocation is computed by the same module that computes the grant - and it
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

    if request.kind == "agent_physical_motion":
        # Cleared rather than set to "0": an absent line lets a stale 1 from a shell profile win the
        # next restart, and a revocation that does not hold across a restart is the worst kind.
        if not str(env.get(_AGENT_MOTION_ENV, "")).strip():
            return {}
        return {_AGENT_MOTION_ENV: ""}

    if request.kind == "teleop_degree_units":
        # Back to the SDK defaults by CLEARING both, not by writing 12.566...:
        # a number frozen here would silently override a future SDK default.
        patch = {}
        if str(env.get(_TELEOP_VALUE_ENV, "")).strip():
            patch[_TELEOP_VALUE_ENV] = ""
        if str(env.get(_TELEOP_SLEW_ENV, "")).strip():
            patch[_TELEOP_SLEW_ENV] = ""
        return patch

    if request.kind == "policy_host_allow":
        host = _host_entry(request.subject)
        if not host:
            return {}
        current = [e.strip() for e in str(env.get(_POLICY_HOST_ENV, "")).split(",") if e.strip()]
        if host not in current:
            # A CIDR entry the operator added by hand may still cover it; say nothing changed
            # rather than narrowing a range this module did not write.
            return {}
        return {_POLICY_HOST_ENV: ",".join(e for e in current if e != host)}

    if request.kind == "policy_type_allow":
        name = request.subject
        if not name:
            return {}
        current = [e.strip() for e in str(env.get(_POLICY_TYPE_ENV, "")).split(",") if e.strip()]
        if name not in current:
            return {}
        return {_POLICY_TYPE_ENV: ",".join(e for e in current if e != name)}

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
