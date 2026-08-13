# Artifact: a renamed path and a capped head side both read as "no shared path"

Measurement for the `check_merge_base_overlap.py --all-open` fix (closes strands-labs/robots#2246).

`capture.py` builds a real git repository, composes the two branches with a real `git merge`,
and runs both the reverted and the current checker over the same topology through a recorded
API seam. `compose.py` draws `rename_overlap.png` and asserts every value it draws against
`facts.json`. Run on Thor, `MUJOCO_GL=egl` (no sim is involved: this is a CI report script).

    PYTHONPATH=<repo> python3 capture.py && python3 compose.py

Headline, all from `facts.json`:

- Branch A renames `pkg/guard.py` to `pkg/limits.py`; branch B adds `CEILING` to
  `pkg/guard.py` and a case importing it there. Alone: `1 passed` and `2 passed`.
- Composed: `git merge` exit `0`, **zero conflicted paths**, `pkg/guard.py` absent,
  `pkg/limits.py` present, and pytest reports
  `ModuleNotFoundError: No module named 'pkg.guard'`.
- The same topology through the checker: `main` exits `0` with no findings; this change
  exits `1` and names `pkg/guard.py` for `#10 + #20`.
- Live queue: 7 open non-draft pull requests, 0 renames, largest changed-file count 10,
  22 requests per sweep against 15 before. Findings unchanged from `main`.
