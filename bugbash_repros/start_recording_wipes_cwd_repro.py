"""Repro: `sim.start_recording(repo_id="", overwrite=True)` wipes the caller's CWD.

Severity: HIGH (destructive data loss on a public, documented API)
Class:    Footgun + Error UX + Missing guard
Target:   v0.5.3  (strands-labs/robots)

Minimal repro — doesn't need lerobot, doesn't need MuJoCo rendering, doesn't
need the dataset recorder to actually init. The destructive write happens
BEFORE any of those steps.

Path through the code:
  strands_robots/simulation/recording.py:1327
      if error := self._already_recording_error(...):  # No repo_id guard above this
          return error
  strands_robots/simulation/recording.py:1342
      dataset_dir = resolve_dataset_dir(repo_id, root)
      # repo_id=""  →  Path(".")                       (strands_robots/dataset_source.py:67,99)
      # repo_id="." →  Path(".")
      # repo_id="/" →  Path("/")
      # repo_id="myproject"  →  Path("myproject")      (joined into CWD)
  strands_robots/simulation/recording.py:1395
      resume_existing = self._prepare_dataset_target(dataset_dir, overwrite)
  strands_robots/simulation/recording.py:1604-1607
      if overwrite:
          if dataset_dir.is_dir():
              shutil.rmtree(dataset_dir)   # ← rm -rf CWD (or /, or ./myproject)

All upstream validators in start_recording() (`fps`, `cameras`, `push_to_hub`,
`overwrite`, `_already_recording_error`) are checked BEFORE this line.
`repo_id` itself has no guard: no empty-string check, no absolute-path check,
no "not CWD" check, no type check. `resolve_dataset_dir` is the writer's
resolution rule and does not guard — that is by design, says its docstring.

Siblings that DO guard their path-ish inputs correctly:
- strands_robots/dataset_source.py:118  (HUB_ID_OUTSIDE_HOME)  catches
  "owner/../etc"  but ONLY on the Hub branch (contains '/'). Any id WITHOUT
  '/'  bypasses it entirely via the local_dataset_dir short-circuit.
- strands_robots/dataset_recorder.py: schema/column validators run AFTER
  rmtree.

Verify (safe): patches `shutil.rmtree` to a spy, confirms the call target.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile


def main() -> int:
    sandbox = tempfile.mkdtemp(prefix="bugbash_cwdwipe_")
    for f in ("important.txt", "secrets.env", "my_code.py"):
        pathlib.Path(sandbox, f).write_text(f"decoy {f}")
    (pathlib.Path(sandbox) / "src").mkdir()
    (pathlib.Path(sandbox) / "src/app.py").write_text("# critical")

    before = sorted(os.listdir(sandbox))
    print(f"Sandbox CWD: {sandbox}")
    print(f"Before:      {before}")

    # Patch shutil.rmtree so we PROVE the call target without actually deleting.
    captured: list[str] = []
    real_rmtree = shutil.rmtree

    def spy(path, *args, **kwargs):
        captured.append(str(path))
        print(f"  !!! shutil.rmtree WOULD HAVE DELETED: {path!r}")
        print(f"      resolves to: {pathlib.Path(path).resolve()}")

    shutil.rmtree = spy  # type: ignore[assignment]
    import strands_robots.simulation.recording as rec_mod

    rec_mod.shutil = shutil  # patched reference inside the module

    os.chdir(sandbox)
    from strands_robots import Robot

    sim = Robot("so101", mesh=False)
    print(f"CWD at start_recording time: {os.getcwd()}")

    # The documented public API, called as a new user would call it after a
    # copy-paste where `repo_id` was unset / defaulted to empty.
    result = sim.start_recording(repo_id="", task="demo", fps=30, overwrite=True)
    print(f"start_recording status: {result.get('status')}")
    msg = result["content"][0]["text"] if result.get("content") else ""
    print(f"  content: {msg[:200]}")

    shutil.rmtree = real_rmtree  # type: ignore[assignment]
    rec_mod.shutil = shutil

    after = sorted(os.listdir(sandbox))
    print(f"After:   {after}")

    # Two possible outcomes, both pinned here:
    #
    # 1. Bug present (pre-fix): shutil.rmtree was called with a path that
    #    resolves to the caller's CWD. Status is "error" from a *later*
    #    stage (dataset-stack probe, schema check, etc.) and the files are
    #    only intact because of the spy. Without the spy they are gone.
    # 2. Bug fixed:            shutil.rmtree is NOT called; start_recording
    #    returns status="error" naming repo_id as the parameter, BEFORE
    #    any directory is touched.
    #
    # Either outcome is a stable pin. A regression would turn the fixed
    # case back into the bug case.
    if captured:
        assert pathlib.Path(captured[0]).resolve() == pathlib.Path(sandbox).resolve(), (
            f"rmtree target was not CWD; got {captured[0]!r}"
        )
        print()
        print(f"BUG PRESENT: shutil.rmtree({captured[0]!r}) resolves to the caller's CWD.")
        print("Without the shutil.rmtree spy, every file above would be gone.")
        _exit = 1
    else:
        assert result.get("status") == "error", (
            f"Expected status='error' from the fix, got {result!r}"
        )
        assert "repo_id" in msg, (
            f"Fix-side refusal message must name repo_id; got {msg!r}"
        )
        print()
        print("BUG FIXED: start_recording refused repo_id='' before any rmtree.")
        _exit = 0

    real_rmtree(sandbox)
    return _exit


if __name__ == "__main__":
    sys.exit(main())
