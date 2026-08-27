"""Turn a safety refusal into something a human can answer."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

# : Same charset the mesh allowlist validator accepts for one entry : (``<org>`` or
# ``<org>/<repo>``).
_HF_ENTRY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$")

_AGENT_TARGET_RE = re.compile(r"starting a task on ([A-Za-z0-9][A-Za-z0-9._:-]{0,63})")
_TRUST_ENV = "STRANDS_TRUST_REMOTE_CODE"
_HF_ENV = "STRANDS_MESH_HF_REPO_ALLOW"
# : The teleop safety envelope: how far one frame may reach, and how fast a joint : may be
# commanded to travel.
_AGENT_MOTION_ENV = "STRANDS_DASH_AGENT_PHYSICAL_MOTION"
_TASK_CONFIRM_ENV = "STRANDS_DASH_TASK_REQUIRES_CONFIRM"
_TELEOP_VALUE_ENV = "STRANDS_MESH_INPUT_VALUE_ABS"
_TELEOP_SLEW_ENV = "STRANDS_MESH_INPUT_SLEW_ABS"
_POLICY_TYPE_ENV = "STRANDS_MESH_POLICY_TYPE_ALLOW"
_POLICY_HOST_ENV = "STRANDS_MESH_POLICY_HOST_ALLOW"
# : The SDK's own charset for an ENTRY in that variable (security._POLICY_HOST_ENTRY_RE),
# copied : rather than imported so this module stays free of the mesh at import time.
_POLICY_HOST_ENTRY_RE = re.compile(r"^[A-Za-z0-9.:/_\-]{1,253}$")
# : Degrees plus a percent gripper: 400 covers a multi-turn wrist with headroom : and still
# refuses a runaway three orders of magnitude out.
_TELEOP_DEGREE_VALUE = "400"
_TELEOP_DEGREE_SLEW = "800"

_PROVIDER_RE = re.compile(r"provider '([^']{1,120})'")
# : Read the value WHOLE - up to the closing quote, or to whitespace when bare - : and
# validate afterwards.
_REPO_QUOTED_RE = re.compile(r"pretrained_name_or_path=(['\"])(.{0,250}?)\1")
_REPO_BARE_RE = re.compile(r"pretrained_name_or_path=([^\s,]{1,250})")
_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# : Taken only from the SDK's fixed `policy_provider=` / `policy_type=` prefix, quoted or
# bare, so a : paraphrase wrapped around the refusal cannot become the subject (the
# _AGENT_TARGET_RE rule).
_POLICY_HOST_SUBJECT_RE = re.compile(
    r"(policy_host|server_address)=(?:'([^']{1,300})'|\"([^\"]{1,300})\"|([^\s,]{1,300}))"
)
_POLICY_SUBJECT_RE = re.compile(r"policy_(?:provider|type)=(?:'([^']{1,80})'|\"([^\"]{1,80})\"|([^\s,]{1,80}))")

#: The refusal text can be a whole traceback; keep the evidence bounded.
_MAX_MESSAGE = 2000

def _host_entry(raw: object, *, strip_url: bool = False) -> str | None:
    """The allowlist ENTRY that grants ``raw``, or None if nothing safe can be derived."""
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
        """Would approving this actually change the environment?"""
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
            # Computed against an EMPTY env deliberately: the question is "is there a grant here at all",
            # not "is it already in place on this machine" - which env_patch answers for the live env at
            # approval time, and which must not disable the button (a refusal from a process started
            # before the last approval still needs its explanation).
            "grantable": self.grantable,
        }

# : The only things an operator can be asked to approve.
KINDS: tuple[str, ...] = (
    "trust_remote_code",
    "hf_repo_allow",
    "teleop_degree_units",
    "agent_physical_motion",
    "policy_type_allow",
    "policy_host_allow",
)

def build_request(kind: str, subject: object = None, message: str = "") -> ConsentRequest | None:
    """Construct a consent request from ``kind`` + ``subject``, validating both."""
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
        # because the gate reads one env var.
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
        # Subject is the arm that was refused, for the dialog's wording only: the envelope is a
        # MACHINE-wide setting, so the scope deliberately is not per-peer.
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
        # One variable, two things the SDK refuses with it.
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
    """Recognise a *continuable* refusal in ``text``, else ``None``."""
    if not isinstance(text, str) or not text.strip():
        return None
    message = text.strip()[:_MAX_MESSAGE]

    if _TRUST_ENV in message:
        m = _PROVIDER_RE.search(message)
        return build_request("trust_remote_code", m.group(1) if m else None, message)

    # The teleop refusal names no env var (it is a per-frame rejection, logged by the follower),
    # so it is recognised by its own words.
    if "input frame value for" in message and "out of range" in message:
        return build_request("teleop_degree_units", None, message)
    if "input frame slew for" in message and "out of range" in message:
        return build_request("teleop_degree_units", None, message)

    if _AGENT_MOTION_ENV in message:
        # The refusal is ours (agent_motion.py), and it names the peer in the same breath as the words
        # MOVE REAL HARDWARE; take the peer only from that fixed prefix so the model's own paraphrase
        # around it cannot become the subject.
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
    """The smallest env change that grants ``request``, given the current ``env``."""
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
    """What this machine currently grants - every kind, in one place."""
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
        # this machine been opened up to", and printing localhost as a grant would bury the one entry
        # that matters among defaults nobody approved.
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
        # tightens it.
        "locks": {
            "task_requires_confirm": (
                str(env.get(_TASK_CONFIRM_ENV, "")).strip().lower() in ("1", "true", "yes", "on")
            ),
            "task_requires_confirm_env": _TASK_CONFIRM_ENV,
        },
    }

def revoke_patch(request: ConsentRequest, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The env change that takes a grant BACK, given the current ``env``."""
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
    """Add ``needs_consent`` to an error ``payload`` if any source is continuable."""
    for source in sources:
        request = classify_refusal(source)
        if request is not None:
            payload["needs_consent"] = request.as_dict()
            break
    return payload
