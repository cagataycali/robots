### Fixed: SPA path traversal is refused at the string level, the numeric-option scope test admits `sim_call`, one `assert` no longer mutates, and a NaN check reads as one

Four defects surfaced on PR #2848 and this change removes each on its own
evidence.

* `strands_robots/dashboard/server.py` — the SPA catch-all's containment
  check was correct at the resolved-path level (`Path.relative_to(_DIST_ROOT)`)
  but CodeQL's `py/path-injection` sink cannot prove a post-resolve guard is
  complete, and the alert stood as three errors on lines 2065/2071/2073. A
  request whose URL segment equals `..` (or whose raw string carries an
  absolute-path leader, a Windows drive prefix or a NUL byte) cannot describe
  a descendant of `_DIST_ROOT`, so the handler now refuses those shapes on the
  string *before* the filesystem is touched and treats the request the way it
  treats a missing file - the SPA entry point is served. The
  `Path.relative_to` guard remains as belt-and-braces so a symlink that points
  outside `dist` still fails closed. Since a legitimate SPA URL never carries
  a `..` segment or an absolute-path leader, no legitimate route breaks.

* `tests/mesh/test_robot_mesh_numeric_option_domain.py` — the scope
  drift-guard asserted `set(_ACTION_NUMERIC_OPTIONS) <= set(ALL_ACTIONS)` but
  the runtime table has carried `sim_call: ("timeout",)` since the
  `robot_mesh` tool grew the `sim_call` verb, so the assertion failed on
  every run with `"sim_call"` in the "extra items in the left set". The
  scope table is correct - `sim_call` really does hand a `timeout` value to
  `Mesh.send`, and refusing an unusable budget there is the same defense
  every other timeout-reading verb has. The fix pins the test's mirror of
  reality: `sim_call` is added to `ALL_ACTIONS` (a real action the tool
  advertises) and to `READS_TIMEOUT` (an action that hands `timeout` to a
  wait, so an unusable value is refused). The pair keeps the two-sided
  scoping test at line 319 non-vacuous - if either table forgets `sim_call`,
  the equality breaks.

* `tests/test_dashboard_peer.py` — the sameness test at line ~225 asserted
  `a.pop("origin") == "managed"` and `b.pop("origin") == "external"` on two
  successive lines, but `dict.pop` is a mutation and `python -O` (which
  drops assertions) would silently skip both rewrites, letting the following
  `assert a == b` see two rows that still carry an `origin` field and
  differ. CodeQL's `py/side-effect-in-assert` reported this on both lines.
  The fix takes the `pop` out of the `assert`: bind `a_origin`, `b_origin`
  first, then compare. The mutation happens whether assertions are enabled
  or not, and the equality check still sees rows without an `origin` field.

* `strands_robots/dashboard/ws_observability.py` — `fps_cap` rejected NaN
  with `v != v`, a legitimate but stylized idiom that CodeQL flagged with
  `py/comparison-of-identical-expressions`. `math.isnan(v)` reads as what it
  is and CodeQL does not flag it. The smoke check
  `fps_cap("nan") is None and fps_cap("-1") is None and fps_cap("5") == 5.0`
  still holds - three independent evidences for the three paths.
