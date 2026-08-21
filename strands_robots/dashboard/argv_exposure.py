"""Was this dashboard's bearer token handed to it on the COMMAND LINE?

`ps` on macOS shows every process's full argv to every local user, so a dashboard started with
``--auth-token <TOKEN>`` publishes its own credential to anyone with a shell on the machine — and the
token is the only thing between a stranger on that box and a form that moves real motors.

This is not a hypothetical: writing RESTART_NOTES.md, an agent read the LIVE token straight out of
`pgrep -fl` output while looking for the dashboard's pid. cli.py already offers the cure
(``--auth-token-file``, which also keeps it out of shell history and refuses to start on an empty
file), so all that was missing was the process noticing its own posture and saying so.

Deliberately quiet about everything else: no token, a token from settings/env, or the file form all
return None. A warning that fires when nothing is wrong is a warning people learn to close.
"""

from __future__ import annotations

FLAG = "--auth-token"


def token_flag_in_argv(argv: list[str] | tuple[str, ...] | None) -> str | None:
    """The exposing argument as it appears in argv, or None.

    Matches ``--auth-token X`` and ``--auth-token=X`` and NOT ``--auth-token-file PATH``: the file
    form is the remedy, and a check that flagged it too would teach the operator to ignore this.
    A flag with no value is not an exposure either - argparse would have refused that start.
    """
    items = list(argv or ())
    for i, arg in enumerate(items):
        if arg == FLAG:
            nxt = items[i + 1] if i + 1 < len(items) else ""
            if nxt and not nxt.startswith("-"):
                return FLAG
        elif arg.startswith(f"{FLAG}="):
            if arg[len(FLAG) + 1:]:
                return FLAG
    return None


def argv_token_notice(argv: list[str] | tuple[str, ...] | None) -> dict[str, str] | None:
    """The sentence for the settings screen, or None when there is nothing to say.

    Names the exposure, who can see it, and the exact flag that fixes it - the same posture as the
    other security disclosures here. It is NOT a refusal: the running dashboard is authenticated and
    working, and rotating a token costs a restart the operator may not want this minute.
    """
    if not token_flag_in_argv(argv):
        return None
    return {
        "kind": "token_in_argv",
        "severity": "warn",
        "text": (
            "This dashboard was started with --auth-token on the command line, so its bearer token "
            "is readable by every local user via `ps` - and that token is what stops a stranger on "
            "this machine from driving the arms."
        ),
        "remedy": (
            "Next restart, pass --auth-token-file ~/.strands_dashboard/local_api_token.txt instead: "
            "same token, same auth, out of argv and out of shell history. Rotate the token if this "
            "machine is shared."
        ),
    }
