"""Mutation table: does each arm catch a plausible regression?

Each anchor is scoped to its enclosing function via AST (several of these
strings appear more than once in the file), and in_fn/in_file counts are
printed as the justification for that scoping.  Sources are restored in a
finally and asserted byte-identical.
"""
import ast, pathlib, subprocess, sys

RENDERING = pathlib.Path("strands_robots/simulation/mujoco/rendering.py")
BASE = pathlib.Path("strands_robots/simulation/base.py")
NEWFILE = "tests/simulation/mujoco/test_render_no_gl_context_message.py"

NEW_TESTS = [
    "test_get_frame_raises_a_clean_message_when_no_gl_context",
    "test_get_frame_raises_rather_than_returning_an_envelope",
    "test_the_compositor_surfaces_the_actionable_message",
    "test_get_world_point_carries_the_actionable_message_into_its_envelope",
    "test_every_renderer_consumer_has_a_pinned_no_gl_channel",
]
K_NEW = " or ".join(NEW_TESTS)
ARM_NEW = ["-k", K_NEW, NEWFILE]
# the suite as it stands: this module's two pre-existing tests plus every other
# module that plausibly covers get_frame / the renderer-None family
ARM_OLD = [
    "-k", f"not ({K_NEW})",
    NEWFILE,
    "tests/simulation/mujoco/test_get_frame_camera_params.py",
    "tests/simulation/mujoco/test_observation_camera_failure_resilience.py",
    "tests/rendering",
]

GET_FRAME_RAISE = '''                raise RuntimeError(
                    "Rendering unavailable (no OpenGL context). "
                    "Install EGL or OSMesa for offscreen rendering: apt-get install libosmesa6-dev"
                )'''

MUTATIONS = [
    ("M1 get_frame returns the envelope instead of raising", RENDERING, "get_frame",
     GET_FRAME_RAISE,
     '''                return {  # type: ignore[return-value]
                    "status": "error",
                    "content": [{"text": "Rendering unavailable (no OpenGL context)."}],
                }'''),
    ("M2 get_frame drops the renderer-None guard", RENDERING, "get_frame",
     '''            if renderer is None:
''' + GET_FRAME_RAISE + '\n',
     ""),
    ("M3 get_frame loses the actionable install hint", RENDERING, "get_frame",
     GET_FRAME_RAISE,
     '''                raise RuntimeError("Rendering unavailable (no OpenGL context).")'''),
    ("M4 get_frame raises a message that names neither GL nor the fix", RENDERING, "get_frame",
     GET_FRAME_RAISE,
     '''                raise RuntimeError("render failed")'''),
    ("M5 get_world_point stops converting the raise to an envelope", BASE, "get_world_point",
     # disambiguated from the identical handler on the get_camera_params call by
     # the comment line that follows only the get_frame one
     """            except (KeyError, ValueError, RuntimeError, TypeError) as e:
                # TypeError included as defense-in-depth for backend lookup""",
     """            except (KeyError, ValueError, TypeError) as e:
                # TypeError included as defense-in-depth for backend lookup"""),
    # A genuinely NEW consumer method -- an extra call inside an existing
    # consumer is not a new channel decision, so it must not (and does not)
    # trip the drift guard.
    ("M6 a fifth renderer consumer ships undecided", RENDERING, "_get_renderer",
     "    def _get_renderer(self, width: int, height: int):",
     """    def peek_pixels(self, width: int = 8, height: int = 8):
        renderer = self._get_renderer(width, height)
        return renderer

    def _get_renderer(self, width: int, height: int):"""),
]


def fn_range(path, fname):
    src = path.read_text()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fname:
            return src, n.lineno, n.end_lineno
    raise AssertionError(f"{fname} not found in {path}")


def run(args):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q", "-p", "no:randomly", "--no-cov", "--timeout=90"],
        capture_output=True, text=True, cwd=".",
    )
    out = r.stdout
    failed = sorted({l.split("::")[-1].split(" ")[0] for l in out.splitlines() if l.startswith("FAILED")})
    tail = [l for l in out.splitlines() if " passed" in l or " failed" in l or "error" in l]
    return failed, (tail[-1].strip() if tail else "?")


originals = {p: p.read_text() for p in (RENDERING, BASE)}
rows = []
try:
    for label, path, fname, old, new in [m for m in MUTATIONS if m[0].startswith('M6')]:
        src, lo, hi = fn_range(path, fname)
        lines = src.splitlines(keepends=True)
        region = "".join(lines[lo - 1:hi])
        in_fn, in_file = region.count(old), src.count(old)
        assert in_fn == 1, f"{label}: anchor appears {in_fn}x inside {fname} (file: {in_file})"
        mutated = src.replace(region, region.replace(old, new, 1), 1)
        assert mutated != src
        ast.parse(mutated)
        path.write_text(mutated)
        try:
            f_new, s_new = run(ARM_NEW)
            f_old, s_old = run(ARM_OLD)
        finally:
            path.write_text(originals[path])
        rows.append((label, in_fn, in_file, len(f_new), s_new, len(f_old), s_old, f_new))
        print(f"{label}\n    anchor in_fn={in_fn} in_file={in_file}")
        print(f"    NEW tests : {len(f_new)} failed  [{s_new}]  {f_new}")
        print(f"    OLD suite : {len(f_old)} failed  [{s_old}]  {f_old}")
finally:
    for p, s in originals.items():
        p.write_text(s)
        assert p.read_text() == s

print("\n=== restore check ===")
print(subprocess.run(["git", "diff", "--stat", "--", str(RENDERING), str(BASE)],
                     capture_output=True, text=True).stdout.strip() or "both sources byte-identical")
print(f"\ncaught by NEW: {sum(1 for r in rows if r[3] > 0)}/{len(rows)}   caught by OLD: {sum(1 for r in rows if r[5] > 0)}/{len(rows)}")
