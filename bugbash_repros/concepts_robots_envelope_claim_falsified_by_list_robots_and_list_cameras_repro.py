"""Repro: docs/concepts/robots.md:26 "Every action returns {status, content}" is
falsified by list_robots() and list_cameras(), both signed `-> list[str]`.

The same claim is repeated three times on the page without any exceptions
clause:

  - Line 2 (frontmatter): "why every method returns the same envelope"
  - Line 24: "## The envelope"
  - Line 26: "Every action returns {status, content} ..."

Sibling page docs/start/first-robot.md:86 was narrowed by harness#691 to name
get_observation() and cleanup() as exceptions, and the follow-up at
harness#736 (open) adds list_robots and list_cameras to that page's narrower
paragraph. The concepts page keeps the unqualified claim and is not touched
by either patch -- so a reader who follows "concepts first, start-page next"
(the order the mkdocs nav presents) hits the broken promise before the fixed
one.

Expected per docs/concepts/robots.md:26:
    robot.list_robots() -> {"status": "success", "content": [...]}
    robot.list_cameras() -> {"status": "success", "content": [...]}

Actual (MuJoCo backend; same shape in Newton/Isaac/mjlab per ABC):
    robot.list_robots() -> ['so101']           (bare list[str])
    robot.list_cameras() -> ['default']         (bare list[str])

A reader following the paragraph literally writes:

    robot.list_robots()["status"]
    # TypeError: list indices must be integers or slices, not str

Verified against upstream strands-labs/robots@main HEAD 5fcc59c (post-harness#758).
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

robot = Robot("so101", mesh=False)

# docs/concepts/robots.md:26 promises this returns {"status", "content"}.
listed_robots = robot.list_robots()
listed_cameras = robot.list_cameras()

assert isinstance(listed_robots, list), (
    f"list_robots returned {type(listed_robots).__name__}, "
    f"not list - repro would need updating. value={listed_robots!r}"
)
assert isinstance(listed_cameras, list), (
    f"list_cameras returned {type(listed_cameras).__name__}, "
    f"not list - repro would need updating. value={listed_cameras!r}"
)

# docs/concepts/robots.md:26 - "Every action returns {status, content}..."
# These two methods are concrete counter-examples; the TypeError below is the
# user-visible symptom when a reader trusts the paragraph literally.
try:
    _ = listed_robots["status"]
    raise AssertionError("list_robots unexpectedly accepted ['status']")
except TypeError as e:
    print(f"BUG CONFIRMED (list_robots): {e}")

try:
    _ = listed_cameras["status"]
    raise AssertionError("list_cameras unexpectedly accepted ['status']")
except TypeError as e:
    print(f"BUG CONFIRMED (list_cameras): {e}")

# Sibling list_* methods DO honour the envelope -- the asymmetry is intrinsic:
listed_objects = robot.list_objects()
listed_bodies = robot.list_bodies()
assert isinstance(listed_objects, dict) and "status" in listed_objects
assert isinstance(listed_bodies, dict) and "status" in listed_bodies
print(f"SIBLING OK (list_objects): status={listed_objects['status']}")
print(f"SIBLING OK (list_bodies):  status={listed_bodies['status']}")

print()
print("docs/concepts/robots.md:26 contradicts list_robots and list_cameras.")
print("Fix: narrow the three paragraphs on docs/concepts/robots.md to name the")
print("same exceptions docs/start/first-robot.md names (per harness#691, #736),")
print("or widen the two listers to the envelope (behavioural; left as B-tier).")
