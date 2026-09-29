---
description: Where the release notes live, what the last three releases changed, and how a change is logged between releases.
---

# Changelog

Release notes live on GitHub; this page is the map to them: where the full notes are, what the last three releases changed, and how a change is logged between releases.

- Full notes: [github.com/strands-labs/robots/releases](https://github.com/strands-labs/robots/releases)
- Assembled file: [`CHANGELOG.md`](https://github.com/strands-labs/robots/blob/main/CHANGELOG.md) in the repository root
- Between releases: every pull request adds one fragment under `changelog.d/`; the file is assembled at tag time with `python scripts/assemble_changelog.py --apply`

The site documents `main` at the commit it was built from, ahead of the newest tag; a feature on these pages missing from your install is in the next release.

## v0.5.2

Released 2026-09-17, 1,526 commits over v0.5.1. The first release that drives real robots without lerobot in the middle: native drivers (`Robot(name, mode="real", driver="strands")`) for Unitree G1 and Go2, Booster T1, Franka, UR, Robotiq, Crazyflie, EarthRover, Reachy Mini, Microduck and the Feetech and Dynamixel buses; the Microduck as a sim and real robot with its own provider; the operator dashboard back (`strands-robots dashboard`) with passkey auth and an operator gate in front of every call that can move hardware; eight security findings closed. Removed: `vera`, `motionbricks`, the vendored LIBERO suite, `lerobot_calibrate`. Floors: `strands-agents` 1.13.0, `eclipse-zenoh` 1.6.1. [Notes](https://github.com/strands-labs/robots/releases/tag/v0.5.2).

## v0.5.1

Released 2026-08-06, 51 commits over v0.5.0. A correctness-only patch: the lerobot floor rose to `>=0.6.1` so `stream_dataset(repo_type="bucket")` resolves to a `StreamingLeRobotDataset` that accepts `repo_type`, and the numeric-input hardening pass continued across training, tools and simulation. [Notes](https://github.com/strands-labs/robots/releases/tag/v0.5.1).

## v0.5.0

Released 2026-08-04, 806 commits over v0.4.1: the NVIDIA Isaac Sim backend, analytic motion primitives (`move_to`, `set_gripper`, `rotate_wrist`), a remote-inference client and server split, terrain locomotion curricula, and a hardening pass over every numeric input the agent and mesh surfaces accept. Changelog assembly moved to per-PR fragments in `changelog.d/`. [Notes](https://github.com/strands-labs/robots/releases/tag/v0.5.0).

## Versioning

The version comes from the nearest release tag (`git describe --match 'v[0-9]*'`); a checkout with no reachable tag builds as `0.1.dev...`, and `git fetch upstream --tags` restores the real number. `strands-robots --version` prints what is installed. The road from 0.5.x to 1.0 is on the [roadmap](../project/roadmap.md).
