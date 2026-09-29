### Added: `docs/hooks/check_sketches.py` verifies every python and bash fence under docs/ statically

`check_fences.py` runs the bare python fences; the 102 titled sketches were never
checked. The new hook parses every fence, resolves imports against the installed
package, checks `Robot(...)` names and keywords against the factory and the class
it builds for that mode and driver, attributes read on bound robots, simulations,
policies, trainers, agents and meshes, tool actions against the published lists,
provider names, extras against `pyproject.toml` and `strands-robots <command>`
lines against their own parsers. A grader
(`tests/test_docs_every_python_fence_names_a_real_api.py`) fails the test job on
any finding. Three snippet defects it and its `--online` mode found are fixed:
the first-policy page's checkpoint sentence, `pip install panda-py` (the
distribution is `panda-python`), and a Hub id that does not exist
(`lerobot/act_base`). All 105 runnable fences pass `check_fences.py` (#4282).
