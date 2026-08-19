"""What the dashboard knows about each camera, and why it cannot see one (U14).

The bug this module exists to kill: ``scan_cameras`` dropped every index it
could not open, so a camera that is *present but unavailable* looked exactly
like a camera that does not exist — it simply vanished from the devices screen.
Live on this machine that meant six real cameras showed up as two bare indices,
and the operator's own words were "our cameras are not visible in the list of
cameras at the moment, they are so critical".

Three separate facts get conflated by a bare probe, and an operator needs all
three separated because the remedy differs:

* **absent** — no such device. Nothing to do.
* **in use** — a running robot (or another app) holds it. Frames exist, they
  just belong to someone else; the fix is to stop that owner, not to replug.
* **blocked** — macOS has not granted camera access to the process running the
  dashboard. This one is invisible in the probe result and lethal: EVERY camera
  fails, and without the reason the natural conclusion is broken hardware.

Everything here is pure. The probing and the OS calls live in
``device_manager``; this module only decides what the payload says.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

#: Probe verdicts, worst-to-best for the UI's sorting purposes.
STATES = ("blocked", "absent", "in_use", "unreadable", "ready")


def classify_probe_stderr(text: str) -> tuple[str, str, str | None]:
    """Turn OpenCV's chatter into (state, reason, remedy).

    OpenCV reports a permission denial and a missing device with the same
    return value (``isOpened() == False``) and tells the truth only on stderr,
    which normally goes to a log nobody reads:

        OpenCV: not authorized to capture video (status 0), requesting...
        OpenCV: camera failed to properly initialize!

    Matching is substring-based and lower-cased on purpose — the exact wording
    varies between OpenCV builds, and a stricter parser would silently fall
    back to "absent", which is the misdiagnosis this whole module exists to
    prevent.
    """
    low = (text or "").lower()
    if "not authorized" in low or "not permitted" in low or "access denied" in low:
        return (
            "blocked",
            "macOS has not granted camera access to the process running the dashboard",
            "System Settings > Privacy & Security > Camera, enable the app that "
            "launched the dashboard (Terminal / VS Code / iTerm), then restart it",
        )
    if "device or resource busy" in low or "busy" in low or "in use" in low:
        return (
            "in_use",
            "another program on this machine is holding the camera",
            "quit the app using it (Zoom, Photo Booth, a stray robot process), then rescan",
        )
    if "failed to properly initialize" in low or "can't open camera" in low or "cannot open" in low:
        # An index that the OS roster does not list is simply not there; an index
        # the roster DOES list, that fails this way, is present and unhappy.
        return ("unreadable", "the camera did not start streaming when opened", "unplug and replug it, then rescan")
    return ("absent", "no camera answered at this index", None)


def _geometry(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    if not entry:
        return {}
    out = {k: entry[k] for k in ("width", "height", "fps") if entry.get(k)}
    return out


def roster_name(index: int, roster: Sequence[Mapping[str, Any]]) -> str | None:
    """Best-effort human name for an index, from the platform listing.

    A GUESS, and labelled as such by the caller: the OS listing order is not
    OpenCV's enumeration order (Continuity cameras renumber themselves, and a
    duplicate model name — this machine has two ``USB2.0_CAM1`` — cannot be
    told apart by name at all). The preview endpoint stays the authority,
    because a picture cannot lie.
    """
    for item in roster:
        if item.get("listing_index") == index:
            name = item.get("name")
            return str(name) if name else None
    return None


def merge_cameras(
    *,
    probed: Iterable[Mapping[str, Any]],
    claimed: Mapping[int, str],
    roster: Sequence[Mapping[str, Any]] = (),
    remembered: Mapping[int, Mapping[str, Any]] | None = None,
    failures: Mapping[int, str] | None = None,
    max_index: int = 4,
) -> list[dict[str, Any]]:
    """One row per camera this machine could plausibly have, never fewer.

    ``probed`` is what opened and read *now*; ``claimed`` maps an index to the
    live peer holding it; ``failures`` maps an index to the stderr its probe
    produced; ``remembered`` is what we last measured for an index while it was
    free. Every index in any of those sources — plus the roster's own indices —
    gets a row, so the count on screen stops depending on who happens to be
    running.

    Geometry is carried over from ``remembered`` for a camera we cannot open
    right now, tagged ``geometry_from: "remembered"``: 1920x1080 measured two
    minutes ago is real information, and blanking it makes a claimed camera
    look broken. Fabricating it as fresh would be worse, hence the tag.
    """
    remembered = remembered or {}
    failures = failures or {}
    probed_by_index = {int(c["index"]): c for c in probed if c.get("index") is not None}

    universe: set[int] = set(probed_by_index) | set(claimed) | set(failures)
    universe |= {int(r["listing_index"]) for r in roster if isinstance(r.get("listing_index"), int)}
    universe |= set(remembered)
    universe = {i for i in universe if 0 <= i}

    rows: list[dict[str, Any]] = []
    for index in sorted(universe):
        row: dict[str, Any] = {"index": index}
        name = roster_name(index, roster)
        if name:
            # `name_is_guess` is not decoration: without it a UI writes the name
            # next to the index as fact, and on this very machine that mislabelled
            # a Logitech as the wrist camera.
            row["name_hint"] = name
            row["name_is_guess"] = True

        if index in probed_by_index:
            row.update(_geometry(probed_by_index[index]))
            row["state"] = "ready"
            row["reason"] = "opened and delivered a frame just now"
        elif index in claimed:
            row["state"] = "in_use"
            row["claimed_by"] = claimed[index]
            row["reason"] = f"streaming for {claimed[index]}"
            row["remedy"] = f"despawn {claimed[index]} to free it"
            row.update(_geometry(remembered.get(index)))
            if _geometry(remembered.get(index)):
                row["geometry_from"] = "remembered"
        else:
            state, reason, remedy = classify_probe_stderr(failures.get(index, ""))
            if state == "absent" and index not in failures and name:
                # The roster lists it but nothing probed it (a scan capped by
                # max_index). Saying "absent" there would be a claim we did not
                # test.
                state, reason, remedy = "unknown", "not probed in this scan", "rescan to test this index"
            row["state"] = state
            row["reason"] = reason
            if remedy:
                row["remedy"] = remedy
            row.update(_geometry(remembered.get(index)))
            if _geometry(remembered.get(index)):
                row["geometry_from"] = "remembered"

        row["available"] = row["state"] == "ready"
        rows.append(row)
    return rows


def blocked_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """A one-line diagnosis for the whole machine, when there is one.

    Per-row reasons are correct but easy to miss when the answer is systemic:
    if nothing could be opened and at least one probe said "not authorized",
    the honest headline is that the *permission* is missing, not that the
    cameras are. Returned separately so the UI can say it once, loudly, instead
    of repeating it on every row.
    """
    if not rows:
        return None
    if any(r.get("state") == "ready" for r in rows):
        return None
    blocked = [r for r in rows if r.get("state") == "blocked"]
    if not blocked:
        return None
    return {
        "kind": "camera_permission",
        "message": (
            "macOS is blocking camera access for the process running the dashboard, "
            "so no camera can be opened here."
        ),
        "remedy": blocked[0].get("remedy"),
        "indices": [r["index"] for r in blocked],
    }
