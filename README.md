# strands-robots — `dev`: v1, built from an empty package

**A `Simulation` is a world. A `Robot` is a body in a world, or on a bus. A `Policy` turns an `Observation` into an `Action`.**

This branch is where v1 is built, one part at a time, starting from nothing. `main` carries v0.5.x until v1.0.0 (Nov 24).

- The design: [#4749](https://github.com/strands-labs/robots/issues/4749)
- The schedule, with a checkbox per part: [#3818](https://github.com/strands-labs/robots/issues/3818)

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/drawings/v08_parts.dark.svg"><img alt="The twelve parts of v1, each one pull request with a gate" src="docs/assets/drawings/v08_parts.paper.svg"></picture>

## Contributing a part

One PR per part, against `dev`, titled in the repo's voice and starting with `Part N of #4749`. The PR carries the part's gate test (red before, green after), a `changelog.d/` fragment, and when it can be seen, an animated drawing (`docs/drawings/_tools/scene.py`) and the robot doing it in MuJoCo.

```bash
pip install hatch
hatch run lint
hatch run test
```
