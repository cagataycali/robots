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
KINDS: tuple[str, ...] = ("trust_remote_code", "hf_repo_allow")


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
