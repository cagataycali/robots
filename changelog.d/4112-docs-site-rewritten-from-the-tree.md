### Docs: the site is written again from the tree, with a MuJoCo viewer on every robot page

The documentation site is rebuilt from the source as it stands: five tabs (Start,
Robots, Learn, Reference, Project), 158 pages, one page per robot generated from
the registry with an in-browser MuJoCo viewer that streams the pinned public model
and lets the reader orbit, move joints and read the `robot.act({...})` call for the
pose. Every count on the site is read from the tree at build time, every runnable
fence is executed by `docs/hooks/check_fences.py`, and every previous URL redirects.
The docs graders under `tests/test_docs_*` are retargeted to the new pages; the
content findings they surfaced are fixed on the docs side in the same change.
