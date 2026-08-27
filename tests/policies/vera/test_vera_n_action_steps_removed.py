"""``n_action_steps`` was a public VERA knob that no code read, so it is deleted.

The field was documented twice as "deploy chunk size (actions executed per
infer)" and consumed nowhere. On ``94de3a5`` its complete set of occurrences
under the shipped provider was six lines - a docstring, a dataclass field, a
``VERA_N_ACTION_STEPS`` environment override, a second docstring, a
``VeraPolicy`` keyword and the line forwarding that keyword into
:class:`~strands_robots.policies.vera.config.VeraConfig` - and not one of them
was a read.

Measured before the deletion, no server and no ``vera`` package:

| spelling | what it did |
| --- | --- |
| ``VeraConfig(n_action_steps=8)`` | stored ``8`` |
| ``VeraConfig(n_action_steps=-7)`` | stored ``-7`` |
| ``VeraConfig(n_action_steps="eight")`` | stored ``'eight'`` |
| ``VeraPolicy(n_action_steps=8)`` | constructed, ``config.n_action_steps == 8`` |
| ``VERA_N_ACTION_STEPS=8`` | stored ``8`` |
| ``build_policy_kwargs("vera", n_action_steps=8)`` | ``{... 'n_action_steps': 8}`` |

and in every row the launched server argv was identical - subprocess mode:

    ['python', '-m', 'vera.server.start_vera_server', '--embodiment', 'mimicgen',
     '--host', '127.0.0.1', '--port', '8800', '--vis-port', '8801',
     '--teacache-thresh', '0.1']

with no ``--n-action-steps`` flag; and docker mode, whose argv is built by a
different class, carrying no ``VERA_N_ACTION_STEPS`` container variable either.
``server_env()`` reported no key containing ``ACTION``.

The chunk length is a server-side quantity: ``_infer`` returns the raw ``[H, D]``
array the server sent and ``_chunk_to_action_dicts`` maps all ``H`` rows into the
queue, so there was no local slicing step for this field to be the width of.

**Why deletion rather than a value domain.** A domain would have refused ``-7``
and then still honored nothing - a knob that validates its input and ignores it,
which is a worse contract than an unvalidated one. That is the reason #2012
settled the neighbouring ``render_width`` on the shared media pixel domain and
deliberately left this field alone: ``render_width`` *is* read, on a hot path,
which is what makes a guard at the config funnel worth having.

**The fourth spelling is the one that makes a half-deletion break at runtime.**
``strands_robots/registry/policies.json`` advertised ``n_action_steps`` in the
``vera`` provider's ``config_keys``, and ``build_policy_kwargs`` forwards exactly
the extra kwargs that appear there. ``VeraPolicy.__init__`` takes no ``**kwargs``,
so a registry entry left behind after the keyword is removed turns
``create_policy("vera", n_action_steps=8)`` from a silently-ignored value into
``TypeError: unexpected keyword argument`` - a regression strictly worse than the
defect. :class:`TestTheRegistryAgreesWithTheConstructor` pins the whole set
rather than this one name, so the same class of half-deletion cannot recur on
another ``vera`` key.

Everything here is offline: no socket, no ``vera`` package, no GPU. The provider
module is imported for its signature only, and the ``msgpack``-dependent
websocket client is never constructed.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies.vera import VeraConfig
from strands_robots.policies.vera.provider import VeraPolicy
from strands_robots.policies.vera.server_runner import make_server_runner
from strands_robots.registry.policies import build_policy_kwargs, get_policy_provider

FIELD = "n_action_steps"
ENV_VAR = "VERA_N_ACTION_STEPS"

#: The shipped VERA provider package - the tree the deleted field lived in.
VERA_PKG = Path(VeraConfig.__module__.replace(".", "/")).parent


def _config(**kwargs: Any) -> VeraConfig:
    """Build a config through the funnel, splatted so a removed name reaches it.

    mypy does not narrow a ``**dict[str, Any]`` splat, which is what lets a test
    pass a keyword the dataclass no longer declares.
    """
    return VeraConfig(**kwargs)


def _launch_argv(mode: str) -> list[str]:
    """Argv the given ``server_mode`` would launch the VERA server with.

    The two modes are built by two different classes reached through
    ``make_server_runner`` - ``VeraServerRunner._build_command`` for a local
    subprocess and ``DockerServerRunner._build_run_command`` for the container -
    so a test that builds only one of them cannot speak for both. Nothing is
    started; only the command is composed.
    """
    runner = make_server_runner(VeraConfig(embodiment="mimicgen", sample_steps=10, server_mode=mode))
    builder = getattr(runner, "_build_command", None) or runner._build_run_command
    return list(builder())


def _vera_sources() -> dict[str, str]:
    """Every ``.py`` file of the shipped VERA provider package, by name."""
    root = Path(__file__).resolve().parents[3] / VERA_PKG
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(root.glob("*.py"))}
    # Non-vacuity: the scan root must actually resolve to the provider package.
    assert "config.py" in files and "provider.py" in files, f"unexpected scan root {root}"
    return files


# --------------------------------------------------------------------------- #
# The three spellings inside the provider
# --------------------------------------------------------------------------- #
class TestTheFieldIsGone:
    def test_the_dataclass_no_longer_declares_it(self):
        """It was ``n_action_steps: int | None = None`` on :class:`VeraConfig`."""
        assert FIELD not in {f.name for f in dataclasses.fields(VeraConfig)}

    def test_a_default_config_carries_no_such_attribute(self):
        """A leftover class attribute would keep ``cfg.n_action_steps`` readable
        even with the dataclass field gone, which is the shape a partial
        deletion takes.
        """
        assert not hasattr(VeraConfig(embodiment="mimicgen"), FIELD)

    @pytest.mark.parametrize("value", [8, -7, 0, "eight", 2.5, None, True])
    def test_the_config_refuses_the_keyword(self, value):
        """Every value the field used to store is now refused by name.

        ``None`` and ``0`` are in the set deliberately: a stale caller passing
        the old default must be told, not silently accepted.
        """
        with pytest.raises(TypeError, match=FIELD):
            _config(embodiment="mimicgen", **{FIELD: value})

    def test_the_policy_refuses_the_keyword(self):
        """It was a ``VeraPolicy`` keyword too, forwarded into the config.

        Python binds arguments before the body runs, so this raises without
        reaching the websocket client - no ``msgpack`` and no socket needed.
        """
        with pytest.raises(TypeError, match=FIELD):
            VeraPolicy(embodiment="mimicgen", **{FIELD: 8})

    def test_it_is_not_a_policy_constructor_parameter(self):
        assert FIELD not in inspect.signature(VeraPolicy.__init__).parameters

    def test_the_environment_override_is_gone(self, monkeypatch):
        """``VERA_N_ACTION_STEPS`` was read in ``__post_init__``.

        A deploy variable that survives the field is the worst leftover: it
        reports nothing at all, on a path nobody constructs by hand.
        """
        monkeypatch.setenv(ENV_VAR, "8")
        assert not hasattr(VeraConfig(embodiment="mimicgen"), FIELD)

    def test_the_name_appears_nowhere_in_the_provider_package(self):
        """The premise of the deletion, asserted against the tree.

        The field's justification was that nothing read it. That is now
        expressible as an absence, which is a stronger property than any
        behavioural assertion about a field that no longer exists: if a later
        change reintroduces the name in any spelling - a docstring promising the
        knob included - this fails and the decision in #2013 gets re-taken
        rather than quietly undone.
        """
        offenders = sorted(name for name, src in _vera_sources().items() if FIELD in src)
        assert offenders == [], f"{FIELD} reappeared in {offenders}"

    def test_the_deploy_variable_appears_nowhere_either(self):
        offenders = sorted(name for name, src in _vera_sources().items() if ENV_VAR in src)
        assert offenders == [], f"{ENV_VAR} reappeared in {offenders}"


# --------------------------------------------------------------------------- #
# The fourth spelling: the policy registry
# --------------------------------------------------------------------------- #
class TestTheRegistryAgreesWithTheConstructor:
    def test_the_key_is_gone_from_the_vera_provider(self):
        assert FIELD not in get_policy_provider("vera")["config_keys"]

    def test_an_extra_kwarg_is_no_longer_forwarded(self):
        """``build_policy_kwargs`` keeps exactly the extras naming a config key.

        Before the registry edit this returned ``{'n_action_steps': 8, ...}``,
        which ``create_policy`` splatted into a constructor that no longer takes
        it.
        """
        assert FIELD not in build_policy_kwargs("vera", **{FIELD: 8})

    def test_every_vera_config_key_is_a_real_constructor_parameter(self):
        """The guard that makes a half-deletion impossible on any ``vera`` key.

        ``VeraPolicy.__init__`` declares no ``**kwargs``, so a ``config_keys``
        entry with no matching parameter is a ``TypeError`` for every caller
        arriving through the factory - and nothing else in the tree compares the
        two. Scoped to ``vera`` because that is the provider this change edits;
        the general form across all providers is #2022.
        """
        params = inspect.signature(VeraPolicy.__init__).parameters
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
            "VeraPolicy grew a **kwargs, which makes a stale registry key silent again "
            "and weakens this guard - reconsider it rather than deleting it"
        )
        orphans = [k for k in get_policy_provider("vera")["config_keys"] if k not in params]
        assert orphans == [], f"config_keys entries with no VeraPolicy parameter: {orphans}"

    def test_the_guard_would_catch_a_planted_orphan(self):
        """Non-vacuity for the test above: an unknown key must be reported."""
        keys = [*get_policy_provider("vera")["config_keys"], "definitely_not_a_parameter"]
        params = inspect.signature(VeraPolicy.__init__).parameters
        assert [k for k in keys if k not in params] == ["definitely_not_a_parameter"]


# --------------------------------------------------------------------------- #
# The knob one line away that *is* wired up stays wired up
# --------------------------------------------------------------------------- #
class TestTheNeighbouringLiveKnobIsUntouched:
    """``sample_steps`` is what "wired up" looks like on this dataclass.

    It sits one line from where ``n_action_steps`` was and is forwarded to the
    server in both launch modes. Pinned here so this deletion is demonstrably a
    removal of the inert field and not of a live one - a test suite that would
    pass with ``sample_steps`` deleted too proves nothing about which field went.
    """

    def test_sample_steps_still_reaches_the_subprocess_argv(self):
        argv = _launch_argv("subprocess")
        assert "--sample-steps" in argv
        assert argv[argv.index("--sample-steps") + 1] == "10"

    def test_sample_steps_still_reaches_the_docker_environment(self):
        assert "VERA_SAMPLE_STEPS=10" in _launch_argv("docker")

    @pytest.mark.parametrize("mode", ["subprocess", "docker"])
    def test_no_launch_path_grew_a_substitute_for_the_deleted_field(self, mode):
        """With the field gone, the remaining check is that neither launch path
        acquired an equivalent under another name.
        """
        argv = _launch_argv(mode)
        assert not [a for a in argv if "action_steps" in a.lower()], argv
