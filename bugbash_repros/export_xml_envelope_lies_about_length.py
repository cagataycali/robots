"""Repro: ``export_xml`` without ``output_path`` returned a success envelope
whose header reported the full XML length but whose body was silently capped
at 2000 chars and ended mid-attribute - round-trip through ``load_scene``
failed with ``XML parse error 7``.

Fire #107 - rotation ``readme_quickstart`` - target v0.5.3.

Upstream site: ``strands_robots/simulation/mujoco/physics.py:3413`` (pre-fix).

Pre-fix shape (bug):

    {"status": "success",
     "content": [{"text": "Model XML (12458 chars):\\n<mujoco ...<texture type=\"2d..."}]}

- ``len(body) == 2003``, 10 455 chars missing with no truncation flag.
- Trailing ``...`` lands mid-attribute, reads as a valid XML continuation.
- Round-trip ``load_scene`` on the inline body: parse error 7 (texture).

Post-fix shape (verifier passes):

    {"status": "success",
     "content": [{"text": "Model XML preview (first 2000 of 12458 chars; "
                          "pass output_path=... for the full MJCF):\\n"
                          "<mujoco ...<!-- truncated -->"}]}

- Header explicitly says ``preview``.
- Body ends with the unambiguous ``<!-- truncated -->`` sentinel.
- A caller feeding the body to a parser sees a comment, not a dangling
  attribute, and can detect the sentinel without string-math on chars counts.

Reproduces with mujoco>=3.5.0 and no extras beyond ``[sim-mujoco]``.

Exit codes:
  0: fixed shape detected
  1: pre-fix bug reproduces
"""

from __future__ import annotations

import pathlib
import re
import sys
import tempfile

sys.path.insert(0, "/home/cagatay/bugbash-strands-robots-1791180027/robots")

from strands_robots import Robot

_PREVIEW_HEADER = re.compile(
    r"^Model XML preview \(first (\d+) of (\d+) chars"
)
_FULL_HEADER = re.compile(r"^Model XML \((\d+) chars\):")


def main() -> int:
    robot = Robot("so100")
    robot.add_object(
        name="red_cube",
        shape="box",
        size=[0.05, 0.05, 0.05],
        position=[0.0, -0.2, 0.025],
        color=[1.0, 0.0, 0.0],
    )
    robot.add_camera(
        name="front",
        position=[0.3, -0.7, 0.45],
        target=[0.0, -0.2, 0.03],
    )

    resp = robot(action="export_xml")
    status = resp["status"]
    text = resp["content"][0]["text"]
    first_line, _, body = text.partition("\n")
    print(f"status      : {status}")
    print(f"header line : {first_line!r}")
    print(f"body length : {len(body)}")
    print(f"body tail   : ...{body[-80:]!r}")

    assert status == "success"

    preview_m = _PREVIEW_HEADER.match(first_line)
    full_m = _FULL_HEADER.match(first_line)

    if preview_m:
        # Post-fix preview shape.
        cap, total = int(preview_m.group(1)), int(preview_m.group(2))
        print(f"\nPost-fix preview shape: cap={cap}, total={total}")
        assert "<!-- truncated -->" in body, (
            "preview header without <!-- truncated --> sentinel"
        )
        assert cap < total, "preview header fires when total <= cap"
        print("Preview envelope is honest.")
        return 0

    if full_m:
        total = int(full_m.group(1))
        if total <= len(body) + 2:
            # Full-inline shape - small scene fits entirely.
            assert body.rstrip().endswith("</mujoco>"), (
                "full-inline body is not well-formed"
            )
            print("\nFull-inline envelope (small scene): well-formed.")
            return 0

        # Pre-fix: header claims a length the body does not deliver.
        print(f"\nPre-fix shape detected: header claims {total} chars, "
              f"body ships {len(body)}.")

        # Prove the round-trip breaks.
        tmp = pathlib.Path(tempfile.mkdtemp()) / "roundtrip.xml"
        tmp.write_text(body[:-3] if body.endswith("...") else body)
        r2 = Robot("so100")
        rr = r2(action="load_scene", scene_path=str(tmp))
        rr_text = rr["content"][0]["text"]
        print(f"round-trip load_scene -> status={rr['status']}, "
              f"text={rr_text[:120]!r}")
        assert rr["status"] == "error" and "XML parse error" in rr_text, (
            "round-trip did not fail with parse error - bug may have changed shape"
        )
        print("REPRO MATCHED: pre-fix bug reproduces (silent-wrong + round-trip-broken).")
        return 1

    print(f"\nUnexpected header shape: {first_line!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
