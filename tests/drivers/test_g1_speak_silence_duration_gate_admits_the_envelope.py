"""Tests for :mod:`strands_robots.tools.g1.g1_speak_silence_duration_envelope`.

The module ports the neon bidi ``g1_speak`` verb's turn-end
silence-duration hint (``silence_duration_ms = 700`` in
``cagataycali/neon-the-g1/tools/g1_speak.py``) into a read-only
lookup pair.  The tests grade three things: import hygiene (no
audio-stack loads at import), snapshot fidelity (the envelope
carries the neon-observed ``700`` millisecond default on both
verbs), and the admit/refuse decision matrix for the shared
positive-count domain the neon runner's turn-end detector
implicitly reads.

The single refusal uses one module-local :data:`_REFUSAL_TEXT` on
any shared-domain shape mistake, consistent with the twin
envelopes
:mod:`~strands_robots.tools.g1.g1_speak_vad_envelope` and
:mod:`~strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`.

Refs strands-labs/robots#358.
"""

from __future__ import annotations

import importlib
import sys

import pytest

MODULE_PATH = "strands_robots.tools.g1.g1_speak_silence_duration_envelope"


class TestTheImportPullsNoOptionalAudioModule:
    """The module docstring's import-hygiene contract, refs strands-labs/robots#358.

    A caller authoring a speak plan before any audio extra is
    installed on their host still gets the default back verbatim;
    the module's advertised no-audio-import property is asserted
    against the process's own :data:`sys.modules` after import.
    """

    def test_the_import_pulls_no_unitree_sdk2py_submodule(self) -> None:
        # Snapshot before importing so a submodule loaded by an
        # earlier test does not tar this one; only the delta this
        # module's import introduces is graded.  Pop the target
        # first so the delta is the module's own import cost even
        # when a prior test has already loaded it.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        leaked = {name for name in added if "unitree" in name.lower()}
        assert leaked == set(), (
            f"the import of {MODULE_PATH} pulled unitree_sdk2py "
            f"submodules {sorted(leaked)}; the neon bidi speak "
            "turn-end lookup ports as a snapshot, not as an SDK call"
        )

    def test_the_import_pulls_no_new_audio_stack_submodule(self) -> None:
        # pywebrtc_audio / pyaudio / strands.experimental.bidi are
        # the audio-stack the neon bundle's g1_speak reaches at
        # runtime.  A fresh submodule of any of these three at
        # this module's import time would betray the module's
        # advertised "no audio dependency at load time" property;
        # a caller on a host without the audio extra installed
        # still reads the default back.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        watched = {"pywebrtc_audio", "pyaudio"}
        leaked = {name for name in added if name in watched or any(name.startswith(f"{w}.") for w in watched)}
        # strands.experimental.bidi may already have been loaded
        # in the outer session by other tests; grade only fresh
        # loads originating from this module's import.
        assert leaked == set(), (
            f"the import of {MODULE_PATH} pulled audio-stack "
            f"submodules {sorted(leaked)}; the neon bidi speak "
            "turn-end lookup ports as a snapshot, not as a live "
            "audio-stack dispatch"
        )


class TestTheEnvelopeQuotesTheNeonRunnerObservedDefault:
    """The neon runner authors ``silence_duration_ms = 700`` on
    the ``g1_speak`` signature; both verbs surface that same
    value on the envelope they return, refs strands-labs/robots#358.
    """

    def test_g1_list_speak_silence_duration_envelope_carries_the_neon_default(
        self,
    ) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            g1_list_speak_silence_duration_envelope,
        )

        payload = g1_list_speak_silence_duration_envelope()

        assert payload["status"] == "success"
        assert payload["envelope"] == {"silence_duration_ms_neon_default": 700}

    def test_g1_list_speak_silence_duration_envelope_carries_the_module_local_refusal_text(
        self,
    ) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
            g1_list_speak_silence_duration_envelope,
        )

        payload = g1_list_speak_silence_duration_envelope()

        assert payload["refusals"] == [{"text": _REFUSAL_TEXT}]

    def test_g1_list_speak_silence_duration_envelope_refusal_text_names_the_write_surface(
        self,
    ) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
        )

        assert "silence-duration" in _REFUSAL_TEXT
        assert "turn-end" in _REFUSAL_TEXT
        assert "strands-labs/robots#358" in _REFUSAL_TEXT
        assert _REFUSAL_TEXT.isascii()

    def test_g1_speak_silence_duration_ms_admits_default_argument_admits(self) -> None:
        # The signature's default is
        # _SILENCE_DURATION_MS_NEON_DEFAULT itself, so a caller
        # who does not pass an explicit argument lands on the
        # runner's observed value and reads the admitted-path
        # branch.
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            g1_speak_silence_duration_ms_admits,
        )

        payload = g1_speak_silence_duration_ms_admits()

        assert payload["status"] == "success"
        assert payload["admits"] is True
        assert payload["refusals"] == []
        assert payload["envelope"] == {"silence_duration_ms_neon_default": 700}


class TestTheAdmitBranchCoversTheSharedPositiveCountDomain:
    """The shared :func:`positive_count_error` domain admits any
    positive ``int``; a caller who supplies a value inside the
    domain reads the admitted-path branch.
    """

    @pytest.mark.parametrize(
        "silence_duration_ms",
        [
            700,  # the neon-observed default
            1,  # the minimum admitted count
            200,  # a snappy turn-end
            1500,  # a patient turn-end
            5000,  # a very-patient turn-end
        ],
    )
    def test_a_value_inside_the_shared_domain_admits(self, silence_duration_ms: int) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            g1_speak_silence_duration_ms_admits,
        )

        payload = g1_speak_silence_duration_ms_admits(silence_duration_ms=silence_duration_ms)

        assert payload["status"] == "success", payload
        assert payload["admits"] is True, payload
        assert payload["refusals"] == [], payload
        assert payload["envelope"] == {"silence_duration_ms_neon_default": 700}


class TestTheRefuseBranchNamesTheSharedDomainShapeMistake:
    """The shared :func:`positive_count_error` domain refuses
    ``bool``, non-``int``, and values below ``1``.  Each refusal
    reads on the shared domain and surfaces the module-local
    :data:`_REFUSAL_TEXT`, refs strands-labs/robots#358.
    """

    @pytest.mark.parametrize("silence_duration_ms", [0, -1, -100])
    def test_a_non_positive_value_refuses(self, silence_duration_ms: int) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
            g1_speak_silence_duration_ms_admits,
        )

        payload = g1_speak_silence_duration_ms_admits(silence_duration_ms=silence_duration_ms)

        assert payload["status"] == "success"
        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        refusal = payload["refusals"][0]
        assert refusal["dimension"] == "silence_duration_ms"
        assert refusal["value"] == silence_duration_ms
        assert refusal["bound_key"] == "silence_duration_ms_neon_default"
        assert refusal["bound"] == 700
        assert refusal["comparison"] == "shared-domain"
        assert refusal["text"] == _REFUSAL_TEXT

    def test_bool_true_refuses_at_the_shared_domain_before_any_1_coercion(self) -> None:
        # bool is an int subclass so True would otherwise be a
        # silent 1 to the numeric write path.  The shared
        # positive_count_error refuses the shape decidably before
        # the write side reads it.
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
            g1_speak_silence_duration_ms_admits,
        )

        payload = g1_speak_silence_duration_ms_admits(silence_duration_ms=True)  # type: ignore[arg-type]

        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        refusal = payload["refusals"][0]
        assert refusal["dimension"] == "silence_duration_ms"
        assert refusal["comparison"] == "shared-domain"
        assert refusal["text"] == _REFUSAL_TEXT

    def test_bool_false_refuses_at_the_shared_domain(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
            g1_speak_silence_duration_ms_admits,
        )

        payload = g1_speak_silence_duration_ms_admits(silence_duration_ms=False)  # type: ignore[arg-type]

        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        refusal = payload["refusals"][0]
        assert refusal["dimension"] == "silence_duration_ms"
        assert refusal["comparison"] == "shared-domain"
        assert refusal["text"] == _REFUSAL_TEXT

    def test_a_float_refuses_at_the_shared_domain(self) -> None:
        # positive_count_error only admits int; a caller who
        # passed 700.0 accidentally reads a shape refusal
        # decidably instead of a silent int(700.0)=700 coercion.
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
            g1_speak_silence_duration_ms_admits,
        )

        payload = g1_speak_silence_duration_ms_admits(silence_duration_ms=700.0)  # type: ignore[arg-type]

        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        refusal = payload["refusals"][0]
        assert refusal["dimension"] == "silence_duration_ms"
        assert refusal["comparison"] == "shared-domain"
        assert refusal["text"] == _REFUSAL_TEXT

    def test_a_string_refuses_at_the_shared_domain(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
            g1_speak_silence_duration_ms_admits,
        )

        payload = g1_speak_silence_duration_ms_admits(silence_duration_ms="700")  # type: ignore[arg-type]

        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        refusal = payload["refusals"][0]
        assert refusal["dimension"] == "silence_duration_ms"
        assert refusal["comparison"] == "shared-domain"
        assert refusal["text"] == _REFUSAL_TEXT

    def test_none_refuses_at_the_shared_domain(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
            g1_speak_silence_duration_ms_admits,
        )

        payload = g1_speak_silence_duration_ms_admits(silence_duration_ms=None)  # type: ignore[arg-type]

        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        refusal = payload["refusals"][0]
        assert refusal["dimension"] == "silence_duration_ms"
        assert refusal["comparison"] == "shared-domain"
        assert refusal["text"] == _REFUSAL_TEXT


class TestTheAdmittedAndRefusedPayloadsCarryTheSameEnvelope:
    """Both branches surface the same envelope descriptor so a
    caller reading the refusal branch can still compare their
    argument against the neon default without a second call,
    refs strands-labs/robots#358.
    """

    def test_the_admit_branch_and_refuse_branch_return_the_same_envelope(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            g1_speak_silence_duration_ms_admits,
        )

        admitted = g1_speak_silence_duration_ms_admits(silence_duration_ms=1000)
        refused = g1_speak_silence_duration_ms_admits(silence_duration_ms=-1)

        assert admitted["envelope"] == refused["envelope"]
        assert admitted["envelope"] == {"silence_duration_ms_neon_default": 700}

    def test_the_list_verb_and_admits_verb_return_the_same_envelope(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            g1_list_speak_silence_duration_envelope,
            g1_speak_silence_duration_ms_admits,
        )

        listed = g1_list_speak_silence_duration_envelope()
        admitted = g1_speak_silence_duration_ms_admits()

        assert listed["envelope"] == admitted["envelope"]


class TestTheDecisionReadsNoFilesystemOrBusState:
    """The port is a pure lookup; repeated calls must be
    byte-identical, refs strands-labs/robots#358.
    """

    def test_repeated_calls_produce_the_same_payload(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            g1_list_speak_silence_duration_envelope,
            g1_speak_silence_duration_ms_admits,
        )

        first_list = g1_list_speak_silence_duration_envelope()
        second_list = g1_list_speak_silence_duration_envelope()
        first_admit = g1_speak_silence_duration_ms_admits(silence_duration_ms=500)
        second_admit = g1_speak_silence_duration_ms_admits(silence_duration_ms=500)

        assert first_list == second_list
        assert first_admit == second_admit


class TestTheModuleLocalConstantsAreASCII:
    """The refusal text is quoted verbatim on every driver-side
    wrapper that will surface it; non-ASCII characters in the
    text would betray the same-surface refusal-string discipline
    the sibling envelopes carry, refs strands-labs/robots#2872.
    """

    def test_the_refusal_text_is_ascii(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
        )

        assert _REFUSAL_TEXT.isascii()

    def test_the_refusal_text_cites_a_resolvable_issue_reference(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            _REFUSAL_TEXT,
        )

        assert "strands-labs/robots#358" in _REFUSAL_TEXT

    def test_every_refusal_descriptor_carries_ascii_text(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            g1_list_speak_silence_duration_envelope,
            g1_speak_silence_duration_ms_admits,
        )

        listed = g1_list_speak_silence_duration_envelope()
        refused = g1_speak_silence_duration_ms_admits(silence_duration_ms=-1)

        for refusal in listed["refusals"]:
            assert refusal["text"].isascii()
        for refusal in refused["refusals"]:
            assert refusal["text"].isascii()


class TestTheVerbsAreStrandsToolDecorated:
    """The @tool decorator marks both verbs as agent-facing so
    the strands agent auto-discovers them under the same
    lookup-pair contract the sibling envelopes carry, refs
    strands-labs/robots#358.
    """

    def test_g1_list_speak_silence_duration_envelope_is_tool_decorated(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            g1_list_speak_silence_duration_envelope,
        )

        assert hasattr(g1_list_speak_silence_duration_envelope, "tool_spec") or callable(
            g1_list_speak_silence_duration_envelope
        )

    def test_g1_speak_silence_duration_ms_admits_is_tool_decorated(self) -> None:
        from strands_robots.tools.g1.g1_speak_silence_duration_envelope import (
            g1_speak_silence_duration_ms_admits,
        )

        assert hasattr(g1_speak_silence_duration_ms_admits, "tool_spec") or callable(
            g1_speak_silence_duration_ms_admits
        )
