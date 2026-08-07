"""Contract pin for what names a GraphQL mutation's subject, and what checks it.

``AGENTS.md`` > PR Workflow > step 8 tells a contributor to verify a pull
request's state by reading it back, and covers the mutation *reporting* the
wrong thing - a ``mergePullRequest`` that says "not mergeable" on a merge that
landed. It did not cover the mutation *addressing* the wrong thing, which is the
more expensive direction because it has no undo.

A mutation names its subject by node ID and by nothing else: ``createIssue``
takes a ``repositoryId``, not an owner and a name. So a well-formed ID that is
wrong does not fail - it succeeds against whatever object it does name. Filing
an issue for this repository with a ``repositoryId`` carried over from an
earlier response rather than queried created issue #1 in an unrelated
third-party repository and returned success. ``deleteIssue`` needs admin on the
*target*, so it could not be undone; it was closed as ``NOT_PLANNED`` with an
apology. See #1916.

What makes the rule worth pinning rather than merely regretting is that the
premise the incident was written up under is measurably false. The ID is *not*
opaque. It is ``<TypePrefix>_<urlsafe-base64(msgpack array)>``, where a
repository is ``[0, databaseId]`` and anything a repository owns is
``[0, repository databaseId, own databaseId]``, so both the type and a
repository ``databaseId`` are readable offline before the write:

=============================  ==========================  =======================
node ID                        decodes to                  target
=============================  ==========================  =======================
``R_kgDORUMiZg``               ``[0, 1162027622]``          this repository
``R_kgDOD1WOFw``               ``[0, 257265175]``           the stray repository
``PR_kwDOD1WOF87DdSjQ``        ``[0, 257265175, ...]``      the same stray one
``PR_kwDORUMiZs7Kw3fA``        ``[0, 1162027622, ...]``     ``uutils/coreutils#11342``
=============================  ==========================  =======================

That third row is the finding the incident write-up did not have: all three
guessed IDs in that run carried **one** wrong repository, so a single stale
value contaminated every mutation. The two that failed did so only because
their own databaseId happened not to exist under that repository
(``Could not resolve to a node``). Failing closed was luck about the guess, not
a property of the API - and the one that got lucky the other way is the one
that wrote.

The fourth row is the correction, and it runs the other way. That ID was handed
to a ``mergePullRequest`` while merging #2006. Its middle field is *this*
repository's ``databaseId``, so a check that reads the target repository out of
the ID clears it - and it resolves to an open pull request in
``uutils/coreutils``, whose repository ``databaseId`` is ``11847500``. The
repository field of an owned object's ID is neither what GitHub routes on nor
validated against the object, so it can name this repository while the object
lives in another. The write was refused by permissions, not by any check.

That makes the decode sound in exactly one direction: a repository that is not
this one is proof of a wrong ID, and a repository that is this one is no
information at all. The rule the guidance now leads with therefore has no decode
in it - read every ID back from a query naming ``owner`` / ``name`` / ``number``
- and the decode is kept as the fast reject it is. See #2007.

So four classes are asserted here, and the executable ones are what keep the
prose ones honest:

``TestTheNodeIdEnvelopeIsCheckableOffline`` *executes* the claim. It decodes
this repository's own node ID and the node IDs of an issue and a pull request in
it, and asserts each recovers the ``databaseId`` the API publishes alongside it
- values obtained from one ``repository(owner: "strands-labs", name: "robots")``
query, which is exactly the literal-owner-and-name query the guidance asks for.
A pin that merely asserted ``AGENTS.md`` *says* the ID is checkable would pass
against a future ID format that had stopped being checkable, leaving the
guidance reading plausibly while advising something impossible. This fails
instead.

``TestTheEnvelopeRepositoryFieldCannotClearAWrite`` *executes* the correction,
against the measured pair rather than against a restatement of it. A test that
only asserted the prose says "proves nothing" would keep passing if the envelope
ever started routing on its repository field, leaving the guidance discouraging a
check that had become sound. This one fails instead, because it holds the
counterexample itself.

``TestTheGuidanceNamesTheDecodableEnvelope`` and
``TestTheGuidanceStatesTheDecodeIsRejectOnly`` pin the prose, because the prose
is the deliverable: an agent reads ``AGENTS.md``, not this module. What is
asserted is *adjacency* rather than vocabulary - the fail-open property, the
decodable envelope and the absence of an undo have to stay in the same breath as
the instruction, since each one alone is unactionable. A future edit tightening
the passage back to "resolve IDs with a query" is exactly the regression, it
looks like an improvement, and nothing else in the tree would notice. That is
the same structural reason ``tests/test_merge_gate_viewer_scope.py`` and
``tests/test_codeql_query_filters.py`` exist, and these text assertions follow
the shape those modules established.

Negative control, re-measured for the #2007 correction. With ``origin/main``'s
``AGENTS.md`` restored (that is, the passage before the correction), the run is
``7 failed, 19 passed``:

- all 7 of ``TestTheGuidanceStatesTheDecodeIsRejectOnly`` fail - the six
  qualifiers and the context guard that locates them;
- all 4 of ``TestTheEnvelopeRepositoryFieldCannotClearAWrite`` pass unchanged,
  because the envelope's behaviour is a property of GitHub's IDs and not of this
  repository's prose. Only the guidance is new;
- the 10 in ``TestTheNodeIdEnvelopeIsCheckableOffline`` and the 5 in
  ``TestTheGuidanceNamesTheDecodableEnvelope`` also pass against the old text, so
  the correction neither invalidated the #1916 pins nor made them depend on it.

The split matters: the executable classes are the ones that would catch the
envelope changing under the advice, and they are deliberately insensitive to how
the advice is worded.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

# Ground truth, read from one `repository(owner: "strands-labs", name: "robots")`
# query. Each pair is a node ID and the `databaseId` the API returns beside it,
# so the decoder below is checked against GitHub's own answer rather than
# against a reimplementation of itself.
_REPOSITORY_NODE_ID = "R_kgDORUMiZg"
_REPOSITORY_DATABASE_ID = 1162027622

#: ``(node ID, own databaseId)`` for objects this repository owns.
_OWNED_OBJECTS = [
    pytest.param("I_kwDORUMiZs8AAAABLT9z4g", 5054100450, id="issue-1916"),
    pytest.param("PR_kwDORUMiZs76PCIu", 4198244910, id="pull-request-1920"),
]

# The node IDs from the #1916 incident. Both name a repository that is not this
# one, and both name the *same* one.
_STRAY_REPOSITORY_NODE_ID = "R_kgDOD1WOFw"
_STRAY_PULL_REQUEST_NODE_ID = "PR_kwDOD1WOF87DdSjQ"
_STRAY_REPOSITORY_DATABASE_ID = 257265175

# The #2007 counterexample. This ID was handed to a `mergePullRequest` while
# merging #2006 and refused only by the token's lack of write permission on a
# stranger's repository. Ground truth for what it resolves to comes from one
# `node(id:) { ... on PullRequest { repository { nameWithOwner databaseId } } }`
# query; it is recorded here as a constant because the pair - a claim of *this*
# repository against an object in another - is the whole of the finding.
_FOREIGN_PULL_REQUEST_NODE_ID = "PR_kwDORUMiZs7Kw3fA"
_FOREIGN_PULL_REQUEST_NUMBER = 11342
_FOREIGN_PULL_REQUEST_REPOSITORY = "uutils/coreutils"
_FOREIGN_PULL_REQUEST_REPOSITORY_DATABASE_ID = 11847500

# The ID that was meant, read back from
# `repository(owner: "strands-labs", name: "robots") { pullRequest(number: 2006) }`
# - the literal-owner-and-name query the guidance asks for, which cannot address
# the wrong repository.
_INTENDED_PULL_REQUEST_NODE_ID = "PR_kwDORUMiZs78G3VE"

# msgpack tags the envelope uses. A GitHub node ID is a short array of unsigned
# integers, so this is the whole grammar needed - anything else is a shape this
# decoder must refuse rather than guess at.
_FIXARRAY_MASK = 0xF0
_FIXARRAY_TAG = 0x90
_UINT32_TAG = 0xCE
_UINT64_TAG = 0xCF
_POSITIVE_FIXINT_MAX = 0x80


def _decode_node_id(node_id: str) -> tuple[str, list[int]]:
    """Split a GitHub node ID into its type prefix and its integer payload.

    Args:
        node_id: A next-generation node ID, ``<TypePrefix>_<base64 payload>``.

    Returns:
        The type prefix (``"R"``, ``"I"``, ``"PR"``, ...) and the decoded
        integers. A repository yields ``[0, databaseId]``; an object a
        repository owns yields ``[0, repository databaseId, own databaseId]``.

    Raises:
        ValueError: If ``node_id`` is not a decodable envelope of that shape.
            Raised rather than returning a partial answer, because a decoder
            that guesses is worse than no decoder here: the value it would
            return is the one used to decide whether a write is safe.
    """
    prefix, _, payload = node_id.partition("_")
    if not payload:
        raise ValueError(f"node ID {node_id!r} has no '_' separator")
    try:
        raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"node ID {node_id!r} payload is not base64: {exc}") from exc
    # base64.urlsafe_b64decode is permissive - it discards bytes outside the
    # alphabet and everything after the padding - so a corrupted payload decodes
    # to the same integers as a valid one. Re-encoding is what proves the whole
    # value was read: "kgDORUMiZg==extra" otherwise yields this repository's id.
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode() != payload.rstrip("="):
        raise ValueError(f"node ID {node_id!r} payload does not round-trip through base64")
    if not raw or raw[0] & _FIXARRAY_MASK != _FIXARRAY_TAG:
        raise ValueError(f"node ID {node_id!r} payload is not a msgpack array")

    values: list[int] = []
    index = 1
    while index < len(raw):
        tag = raw[index]
        if tag == _UINT32_TAG:
            (value,) = struct.unpack_from(">I", raw, index + 1)
            index += 5
        elif tag == _UINT64_TAG:
            (value,) = struct.unpack_from(">Q", raw, index + 1)
            index += 9
        elif tag < _POSITIVE_FIXINT_MAX:
            value = tag
            index += 1
        else:
            raise ValueError(f"node ID {node_id!r} holds unsupported msgpack tag {tag:#x}")
        values.append(value)
    expected = raw[0] & ~_FIXARRAY_MASK
    if len(values) != expected:
        raise ValueError(f"node ID {node_id!r} declares {expected} values, decoded {len(values)}")
    return prefix, values


def _claimed_repository(node_id: str) -> int:
    """The repository ``databaseId`` that ``node_id`` *claims*, decoded offline.

    A repository's own ID names itself directly and authoritatively. For anything
    it owns, this is only a claim: the value is carried in the envelope but is not
    what GitHub resolves the object by, and
    ``TestTheEnvelopeRepositoryFieldCannotClearAWrite`` holds an ID whose claim is
    this repository and whose object is in ``uutils/coreutils``. So a mismatch
    disproves an ID and a match establishes nothing - the name says ``claimed``
    rather than ``target`` because reading it as the target is the #2007 defect.
    """
    prefix, values = _decode_node_id(node_id)
    if prefix == "R":
        return values[1]
    if len(values) < 3:
        raise ValueError(f"node ID {node_id!r} names no repository")
    return values[1]


class TestTheNodeIdEnvelopeIsCheckableOffline:
    """The type and the target repository are readable before the write."""

    def test_the_repository_id_decodes_to_its_published_database_id(self) -> None:
        prefix, values = _decode_node_id(_REPOSITORY_NODE_ID)
        assert prefix == "R"
        assert values == [0, _REPOSITORY_DATABASE_ID], (
            "This repository's node ID no longer decodes to the databaseId the API "
            "publishes beside it. Either the envelope format changed - in which case "
            "the AGENTS.md guidance that a repositoryId can be checked offline is now "
            "wrong and must be corrected rather than this test relaxed - or the "
            "constants drifted. See #1916."
        )

    @pytest.mark.parametrize(("node_id", "database_id"), _OWNED_OBJECTS)
    def test_an_owned_object_carries_the_repository_it_belongs_to(self, node_id: str, database_id: int) -> None:
        _, values = _decode_node_id(node_id)
        assert values == [0, _REPOSITORY_DATABASE_ID, database_id], (
            f"{node_id!r} should decode to [0, this repository, its own databaseId]. "
            "That middle element is what lets a mutation on an issue or a pull "
            "request be checked against the repository it was meant for."
        )

    def test_the_type_prefix_separates_a_repository_from_what_it_owns(self) -> None:
        # A `PR_...` handed to a parameter wanting a `repositoryId` is wrong by
        # type alone, with nothing else to consult - the cheapest of the checks.
        assert _decode_node_id(_REPOSITORY_NODE_ID)[0] == "R"
        assert _decode_node_id("PR_kwDORUMiZs76PCIu")[0] == "PR"

    def test_the_stray_id_is_distinguishable_from_the_intended_one(self) -> None:
        # The check that would have caught #1916, in the form it was available:
        # the two spellings are visually close and decode to different targets.
        assert _claimed_repository(_REPOSITORY_NODE_ID) == _REPOSITORY_DATABASE_ID
        assert _claimed_repository(_STRAY_REPOSITORY_NODE_ID) == _STRAY_REPOSITORY_DATABASE_ID
        assert _claimed_repository(_STRAY_REPOSITORY_NODE_ID) != _claimed_repository(_REPOSITORY_NODE_ID)

    def test_every_stray_id_from_the_incident_names_one_wrong_repository(self) -> None:
        strays = {_STRAY_REPOSITORY_NODE_ID, _STRAY_PULL_REQUEST_NODE_ID}
        targets = {_claimed_repository(node_id) for node_id in strays}
        assert targets == {_STRAY_REPOSITORY_DATABASE_ID}, (
            "The repository ID and the pull-request ID guessed in that run should both "
            "decode to the same wrong repository. That is why one stale value was able "
            "to contaminate three mutations, and why the two that failed failed only by "
            "luck about their own databaseId rather than by any check. See #1916."
        )

    @pytest.mark.parametrize(
        "malformed",
        ["RkgDORUMiZg", "R_!!!!", "R_AAAA", "R_kgDORUMiZg==extra"],
        ids=["no-separator", "not-base64", "not-an-array", "trailing-garbage"],
    )
    def test_a_shape_it_cannot_read_is_refused(self, malformed: str) -> None:
        # Non-vacuity: the decoder must not answer for an envelope it cannot
        # read. A plausible-looking integer here would be used to decide that a
        # write is safe, so guessing is worse than refusing.
        with pytest.raises(ValueError):
            _decode_node_id(malformed)


class TestTheEnvelopeRepositoryFieldCannotClearAWrite:
    """A decode showing the right repository is no evidence about the target.

    ``TestTheNodeIdEnvelopeIsCheckableOffline`` above establishes that the
    envelope is readable. This class establishes what reading it does *not* buy,
    which is the half #1916's write-up asserted and #2007 measured.
    """

    def test_the_foreign_id_claims_this_repository(self) -> None:
        assert _claimed_repository(_FOREIGN_PULL_REQUEST_NODE_ID) == _REPOSITORY_DATABASE_ID, (
            f"{_FOREIGN_PULL_REQUEST_NODE_ID!r} should decode to a middle field equal to this "
            "repository's databaseId. That is what makes it a counterexample rather than a "
            "curiosity: the pre-write decode AGENTS.md used to prescribe reads this value and "
            "clears the write. See #2007."
        )

    def test_the_object_it_claims_this_repository_for_is_in_another_one(self) -> None:
        # The claim is not merely unauthoritative, it is false about the object:
        # the pull request's real repository databaseId is a different number.
        assert _FOREIGN_PULL_REQUEST_REPOSITORY_DATABASE_ID != _REPOSITORY_DATABASE_ID
        assert _claimed_repository(_FOREIGN_PULL_REQUEST_NODE_ID) != _FOREIGN_PULL_REQUEST_REPOSITORY_DATABASE_ID, (
            f"{_FOREIGN_PULL_REQUEST_NODE_ID!r} resolves to {_FOREIGN_PULL_REQUEST_REPOSITORY} "
            f"#{_FOREIGN_PULL_REQUEST_NUMBER}, whose repository databaseId is "
            f"{_FOREIGN_PULL_REQUEST_REPOSITORY_DATABASE_ID}. If the envelope has started "
            "carrying the object's real repository, the field has become authoritative and the "
            "AGENTS.md guidance that a matching decode proves nothing is now too strict - "
            "correct the guidance rather than relaxing this test. See #2007."
        )

    def test_the_decode_disproves_an_id_but_never_confirms_one(self) -> None:
        # Both directions in one assertion pair, so neither can rot alone.
        # Reject: the #1916 stray claims a repository that is not this one.
        assert _claimed_repository(_STRAY_REPOSITORY_NODE_ID) != _REPOSITORY_DATABASE_ID
        # No clearance: a claim equal to this repository holds for an ID that
        # names an object here and for one that does not, so the two are
        # indistinguishable by decode alone.
        assert _claimed_repository(_INTENDED_PULL_REQUEST_NODE_ID) == _claimed_repository(
            _FOREIGN_PULL_REQUEST_NODE_ID
        ), (
            "The intended ID and the foreign one should be indistinguishable by their decoded "
            "repository field. That indistinguishability is the reason the guidance leads with "
            "reading an ID back rather than with decoding it. See #2007."
        )

    def test_a_prefix_comparison_against_a_known_good_id_also_clears_it(self) -> None:
        # The check that actually happened, and it is the same unsound test done
        # less precisely: eyeballing the ID against a known-good one for this
        # repository. They agree on 14 of 19 characters.
        shared = 0
        for left, right in zip(_INTENDED_PULL_REQUEST_NODE_ID, _FOREIGN_PULL_REQUEST_NODE_ID):
            if left != right:
                break
            shared += 1
        assert shared == 14
        assert len(_FOREIGN_PULL_REQUEST_NODE_ID) == len(_INTENDED_PULL_REQUEST_NODE_ID) == 19, (
            "Equal lengths and a 14-character shared prefix are what make the two IDs "
            "interchangeable to a reader. Recorded so the guidance's claim about eyeballing is "
            "measured rather than asserted. See #2007."
        )


def _agents_text() -> str:
    return _AGENTS_PATH.read_text(encoding="utf-8")


#: The sentence the correction introduces. Every other assertion is positioned
#: from it, so its absence fails outright rather than making the rest vacuous.
_SUBJECT_CLAIM = "names its subject by node ID"

#: How far a qualifier may sit from the claim while still reading as one
#: instruction. Generous enough to survive rewording, tight enough that moving a
#: qualifier out of step 8 fails.
_ADJACENCY_WINDOW = 2600


def _window_after(text: str, anchor: str) -> str | None:
    """The ``_ADJACENCY_WINDOW`` characters following ``anchor``, or ``None``."""
    position = text.find(anchor)
    if position < 0:
        return None
    return text[position : position + _ADJACENCY_WINDOW]


class TestTheGuidanceNamesTheDecodableEnvelope:
    """The rule is only actionable with all three qualifiers beside it."""

    def test_the_subject_claim_is_present(self) -> None:
        # Context guard: the assertions below are positioned from this phrase, so
        # a silent rewording would move the pin rather than break it.
        assert _SUBJECT_CLAIM in _agents_text(), (
            f"AGENTS.md no longer contains {_SUBJECT_CLAIM!r}, which this class uses to "
            "locate the node-ID rule. If the claim was deliberately reworded, update "
            "_SUBJECT_CLAIM to match rather than deleting these tests - the point is "
            "that the rule and its qualifiers stay together."
        )

    def test_the_guidance_states_that_a_wrong_id_fails_open(self) -> None:
        window = _window_after(_agents_text(), _SUBJECT_CLAIM)
        assert window is not None and "does not fail" in window, (
            "AGENTS.md must say that a well-formed but wrong node ID succeeds against "
            "whatever object it does name. Without that, the rule reads as tidiness "
            "rather than as the reason the write is unsafe. See #1916."
        )

    def test_the_guidance_names_the_decodable_envelope(self) -> None:
        window = _window_after(_agents_text(), _SUBJECT_CLAIM)
        assert window is not None and "databaseId" in window, (
            "AGENTS.md must say that a node ID decodes to a type and a target "
            "repository databaseId offline. 'Always query the ID' is advice that can be "
            "forgotten under a stale value, which is exactly what happened; a check that "
            "can be run on the value in hand is not. See #1916."
        )

    def test_the_guidance_states_that_there_is_no_undo(self) -> None:
        window = _window_after(_agents_text(), _SUBJECT_CLAIM)
        assert window is not None and "deleteIssue" in window, (
            "AGENTS.md must say that a write to the wrong repository cannot be undone - "
            "deleteIssue needs admin on the target. That is what makes this a "
            "check-before rather than a verify-after. See #1916."
        )

    def test_the_guidance_tells_the_reader_to_check_the_response(self) -> None:
        window = _window_after(_agents_text(), _SUBJECT_CLAIM)
        assert window is not None and "url" in window, (
            "AGENTS.md must keep the response-url check beside the rule: it is the only "
            "signal for the cases the envelope cannot cover, and in #1916 it was the "
            "single clue that anything had gone wrong. See #1916."
        )


#: The correction's own claim. Positioned like ``_SUBJECT_CLAIM`` above, and
#: asserted first so the rest of the class cannot pass vacuously.
_REJECT_ONLY_CLAIM = "fast reject and never a clearance"


def _squash(text: str) -> str:
    """Collapse whitespace runs, so a pin survives a rewrap of the same prose."""
    return " ".join(text.split())


def _bullet_after(text: str, anchor: str) -> str | None:
    """The step-8 bullet containing ``anchor``, whitespace-squashed.

    ``TestTheGuidanceNamesTheDecodableEnvelope`` measures adjacency with a fixed
    ``_ADJACENCY_WINDOW``, which suited a passage that fit inside it. The #2007
    correction roughly tripled this bullet's length, and a fixed window then has
    to be either retuned on every edit or long enough to reach into the *next*
    bullet - at which point it stops testing adjacency at all. The bullet
    boundary is the unit the claim is actually about ("in the same breath as the
    instruction"), it needs no constant, and moving a qualifier to a neighbouring
    bullet still fails. ``\n   - *`` is the marker step 8's bullets begin with.
    """
    start = text.find(anchor)
    if start < 0:
        return None
    end = text.find("\n   - *", start)
    return _squash(text[start:end] if end > 0 else text[start:])


class TestTheGuidanceStatesTheDecodeIsRejectOnly:
    """The decode's one-directionality, the read-back rule, and the asymmetry.

    Each of these is what makes the instruction beside it actionable, and each
    would read as a reasonable simplification to delete. A future edit restoring
    "the target repository is readable before the write" is exactly the #2007
    regression, it looks like a tightening, and nothing else in the tree notices.
    """

    def test_the_reject_only_claim_is_present(self) -> None:
        # Context guard: every assertion below is located from this phrase.
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and _REJECT_ONLY_CLAIM in bullet, (
            f"AGENTS.md must state that the node-ID decode is a {_REJECT_ONLY_CLAIM!r}. If the "
            "wording changed deliberately, update _REJECT_ONLY_CLAIM rather than deleting these "
            "tests - the point is that the one-directionality stays stated. See #2007."
        )

    def test_the_guidance_says_a_matching_decode_proves_nothing(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "proves nothing" in bullet, (
            "AGENTS.md must say that a decode showing this repository proves nothing. Without it "
            "the decode reads as a pre-write safety check, which is the false safe #2007 "
            "measured: an ID claiming this repository merged against uutils/coreutils."
        )

    def test_the_guidance_forbids_constructing_or_carrying_an_id(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "never construct or carry over a node ID" in bullet, (
            "AGENTS.md must state the rule that has no decode in it. Both #1916 and the #2007 "
            "reproductions began with an ID that was constructed or carried rather than queried, "
            "so this is the sentence that prevents the class, not the decode."
        )

    def test_the_guidance_names_the_read_back_query_shape(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "`owner` / `name` / `number`" in bullet, (
            "AGENTS.md must name the query shape that cannot address the wrong repository. "
            "'Query the ID' is ambiguous between `node(id:)` - which takes the ID already in "
            "doubt - and a literal owner-and-name lookup, and only the second is sound."
        )

    def test_the_guidance_carries_the_measured_counterexample(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None
        assert _FOREIGN_PULL_REQUEST_NODE_ID in bullet, (
            "AGENTS.md must carry the ID that clears the decode and resolves elsewhere. A rule "
            "stated without it is advice a future reader may reasonably re-derive their way out "
            "of; the ID is the thing that cannot be argued with. See #2007."
        )
        assert str(_FOREIGN_PULL_REQUEST_REPOSITORY_DATABASE_ID) in bullet, (
            "AGENTS.md must record the resolved repository's real databaseId beside the claimed "
            "one. Two different numbers for one object is the whole argument that the envelope's "
            "repository field is not what GitHub routes on."
        )

    def test_the_guidance_states_the_create_merge_asymmetry(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "not symmetric" in bullet, (
            "AGENTS.md must say the exposure is not symmetric between a merge and a create. The "
            "#2007 merge was refused by permissions and left nothing behind; the createIssue "
            "reproduction succeeded and could not be deleted. Only the second is irreversible, "
            "and that is what should change behaviour at the call site."
        )

    def test_the_guidance_records_the_remedy_for_a_landed_write(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "retitle" in bullet, (
            "AGENTS.md must record what to do about a stray write that already landed. The "
            "instinct is deleteIssue, which is refused for want of admin on the target, so "
            "without the remedy written down the incident stays as expensive as it was."
        )
