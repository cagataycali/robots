"""Repro: README quickstart `list_objects` has no json block (MuJoCo + Newton).

The README (strands-labs/robots@README.md:51-57) sets up the hero scene with
`robot.add_object(name="red_cube", ...)` and `Agent(tools=[robot])("pick up the
red cube")`. A programmatic verifier (CI check, dataset recorder, or an agent
that wants to confirm the add took) calls `robot.list_objects()`.

- MuJoCo backend returns envelope `{status, content:[{text}]}` — ONE content
  item, text-only. No `json` block.
- Newton backend returns the same text-only envelope.
- Isaac backend returns `{status, content:[{text}, {json:{objects:{...}}}]}` —
  a structured machine-readable block.
- The sibling `list_bodies` on MuJoCo *does* carry a `json` block
  (`{"bodies": [...], "gripper_body": ...}`), so the two neighbouring discovery
  surfaces on the SAME backend disagree on the envelope's second half.

A caller cannot read object names, shapes, positions, static flags
programmatically on MuJoCo or Newton without regex-parsing the text block.

Run:
    MUJOCO_GL=egl python bugbash_repros/list_objects_missing_json_repro.py
"""

from __future__ import annotations

import os
import sys

os.environ.pop("SYSTEM_PROMPT", None)
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot


def shape_report(name: str, envelope: dict) -> tuple[int, bool, bool]:
    content = envelope.get("content", [])
    return (
        len(content),
        any("text" in c for c in content),
        any("json" in c for c in content),
    )


def main() -> int:
    r = Robot("so100", mesh=False)
    # Exactly the README's hero add_object call, twice over.
    r.add_object(
        name="red_cube",
        shape="box",
        size=[0.05, 0.05, 0.05],
        position=[0.0, -0.2, 0.025],
        color=[1.0, 0.0, 0.0],
    )
    r.add_object(
        name="blue_cube",
        shape="box",
        size=[0.05, 0.05, 0.05],
        position=[0.1, -0.2, 0.025],
        color=[0.0, 0.0, 1.0],
    )

    # Direct method call (what a notebook / script uses).
    lo_direct = r.list_objects()
    lb_direct = r.list_bodies()
    # Tool-dispatcher call (what an Agent uses: Robot(action="list_objects")).
    lo_disp = r(action="list_objects")

    probes = [
        ("list_objects (direct)", lo_direct),
        ("list_objects (dispatcher)", lo_disp),
        ("list_bodies (direct)   [sibling, same backend]", lb_direct),
    ]
    failures: list[str] = []
    for label, env in probes:
        items, has_text, has_json = shape_report(label, env)
        marker = "OK " if has_json else "BAD"
        print(f"  {marker}  {label:48s}  items={items}  text={has_text}  json={has_json}")
        if "list_objects" in label and not has_json:
            failures.append(label)

    # Isaac is text+json by design (see strands_robots/simulation/isaac/introspection.py:159):
    print("\n  REF  isaac/introspection.py:list_objects         json={'objects': {name: {...}}}  [has json]")

    # Prove the asymmetry is user-visible: text has to be regex-parsed to recover
    # a structured record. list_bodies on the same backend already hands you
    # the parsed form.
    print("\n  Raw text (what MuJoCo/Newton list_objects gives a programmatic caller):")
    print("    " + repr(lo_direct["content"][0]["text"]))
    print("\n  Compare: list_bodies.json keys (same backend, sibling discovery op):")
    json_block = next((c["json"] for c in lb_direct["content"] if "json" in c), {})
    print("    " + repr(sorted(json_block.keys())))

    if failures:
        print(
            "\n  FAIL  list_objects on MuJoCo returns no 'json' block; "
            "sibling list_bodies does and Isaac's list_objects does. "
            "This asymmetry forces a text-regex parse on the README's primary "
            "verification path for add_object()."
        )
        return 1
    print("\n  OK    list_objects now carries a json block on MuJoCo (defect fixed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
