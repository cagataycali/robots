### Fixed: a narrowed dashboard session window is honoured or refused, never widened

`STRANDS_DASH_AUTH_TOKEN_TTL`, `..._SESSION_MAX_AGE` and `..._HANDOFF_TTL` are how
an operator shortens the window in which a dashboard session can command real
hardware: how long a session token lives, the absolute age past which no renewal
extends it, and the lifetime of a token that rides in a LAN URL. Each was read
with a bare `int(os.getenv(...))` under `except ValueError: return <the default>`.

So every spelling that is not a plain integer resolved to the shipped default,
and the shipped default is the WIDER value in all three cases. Measured end to
end through the surface that spends each one, `TOKEN_TTL=1h` produced a 86400
second session, `SESSION_MAX_AGE=1h` a 30-day renewal cap, and `HANDOFF_TTL=30s`
a 300 second URL token - 24x, 720x and 10x what was asked for, with nothing
logged. `15m`, `3600s`, `1 hour` and a trailing `# comment` all behaved the same
way. Values at or below zero parsed cleanly instead, and mint tokens that are
already expired when handed out, so nobody can sign in.

The module already stated the rule this broke. `_challenge_cap` refuses a cap it
cannot use precisely so that "an operator who narrowed a cap and mistyped it must
hear about it, not silently be handed the wide default back", and `auth_enabled`
reports an unrecognized boolean rather than letting a typo drop the guard. The
duration knobs are now read through one `_duration()` domain with the same
posture, resolved once at import so a misspelling stops the server rather than
first surfacing as a failed login on a dashboard that is already serving. Their
defaults live in one `_DURATION_DEFAULTS` table, so no reader can hand back a
number the documentation does not state.

An empty variable still means unset, whitespace and `3_600` are still accepted,
and a maximum age below the token lifetime is still a legal pair - unlike the
challenge caps, no cross-knob rule is imposed, because "tokens last a day but
re-authenticate every hour" is a coherent request and the cap simply wins.
