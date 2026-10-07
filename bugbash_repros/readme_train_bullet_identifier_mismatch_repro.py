"""README `## What you get` → **Train** bullet names trainer brands as if they
were drop-in `create_trainer()` arguments. None of the four names it writes
work as inputs to the factory the same section links to.

Reproduces harness#TBD. Target: v0.5.3.

README `## What you get` → **Train** bullet (``README.md:98``):

    | **Train** with LeRobot (ACT to GR00T N1.7), Cosmos 3 or RL
    (PPO / FastSAC), locally or as a SageMaker job, ... | [Training](...) |

The four non-"RL" tokens the row writes - **LeRobot**, **Cosmos 3**, **PPO**,
**FastSAC** - look like identifiers (title-case brand next to a parenthetical).
The link under the row is the Training index whose first page sample at
``docs/learn/training/index.md:34`` is a Python one-liner::

    >>> list_trainers()
    ['cosmos3', 'fast_sac', 'fast_td3', 'isaaclab', 'lerobot_local', 'mock', 'ppo', 'sagemaker']

Every single one of the four README-row tokens is wrong at that function:

* ``LeRobot``   -> ``lerobot_local``  (the row drops ``_local``)
* ``Cosmos 3``  -> ``cosmos3``        (the row inserts a space)
* ``PPO``       -> ``ppo``            (the row upper-cases)
* ``FastSAC``   -> ``fast_sac``       (the row drops the ``_`` AND camel-cases)

``create_trainer`` is the ONE resolver behind ``list_trainers`` and both the
Training index and ``docs/learn/training/rl.md:77`` use the snake_case forms as
identifiers while using the row's CamelCase forms as prose. The README row is
the ONLY place in the repository that writes any of the four as a bare
identifier without a snake_case form naming the actual argument.

Only the ``create_trainer("<token>")`` failure path carries a Did-you-mean
hint (factory.py:287). A user who treats the row as the authoritative list
of trainers advertised by this package - the way the sibling rows for sim
(``[sim-mujoco]`` on L87) and extras (``lerobot`` / ``groot`` / ``cosmos3-service``
on L74-75) literally spell their arguments - never calls the factory.
The row's brand-list reads like the matrix at
``docs/learn/training/index.md:52`` which lists ``ppo``, ``fast_sac``,
``fast_td3`` as the exact identifiers.

Fix: 1-sentence rewrite of README.md:98 that pairs each brand with the token
that ``create_trainer()`` accepts, matching the style of
``docs/learn/training/rl.md:2``:

    > "the PPO, FastSAC and FastTD3 trainers"
    (prose) ... combined with
    > "the create_trainer('ppo' | 'fast_sac' | 'fast_td3') trainers"
    (code) in ``docs/learn/training/rl.md:112``.

The did-you-mean path is correct and does NOT need to change; this is a 1-row
docs-only edit on README.md:98 pinning the identifier the user is nudged to
on a 404.
"""

from __future__ import annotations

import os

# Thor shell wedge discipline: scrub SYSTEM_PROMPT to keep env clean.
os.environ.pop("SYSTEM_PROMPT", None)

from strands_robots.training.factory import create_trainer, list_trainers


def main() -> int:
    print(f"available trainers (what the Training page's quickstart prints):")
    print(f"  {list_trainers()}\n")

    # The four tokens the README `## What you get` Train bullet writes.
    README_TRAIN_BULLET_TOKENS = [
        ("LeRobot", "lerobot_local"),  # README L98 writes `LeRobot`
        ("Cosmos 3", "cosmos3"),  # README L98 writes `Cosmos 3` (with space!)
        ("PPO", "ppo"),  # README L98 writes `PPO`
        ("FastSAC", "fast_sac"),  # README L98 writes `FastSAC`
    ]

    all_rejected = True
    for readme_token, actual_identifier in README_TRAIN_BULLET_TOKENS:
        try:
            create_trainer(readme_token)
            print(f"  {readme_token!r:12} ✓ (unexpected - this should fail!)")
            all_rejected = False
        except ValueError as e:
            msg = str(e)
            cites_correct = f"'{actual_identifier}'" in msg
            print(
                f"  {readme_token!r:12} ✗ ValueError, hints {actual_identifier!r}? {cites_correct}"
            )
        except Exception as e:
            print(f"  {readme_token!r:12} ✗ {type(e).__name__}: {e!s:.120}")

    # Prove the actual snake_case tokens DO work (so the only thing wrong is the
    # label the README writes - no backend is missing).
    print("\nActual identifiers from `list_trainers()` do work:")
    for identifier in ("lerobot_local", "cosmos3", "ppo", "fast_sac"):
        try:
            t = create_trainer(identifier)
            print(f"  {identifier!r:16} ✓ {type(t).__name__}")
        except Exception as e:
            # Trainer dependency might be missing on a minimal install - that
            # is a SEPARATE concern; here all we care about is that the name
            # resolves (does not raise ValueError about unknown provider).
            msg = str(e)
            is_name_error = "No trainer registered" in msg
            print(
                f"  {identifier!r:16} {'✗' if is_name_error else '✓ (deps-only)'} "
                f"{type(e).__name__}: {msg[:120]}"
            )

    # Exit code: 1 iff any of the README tokens happens to work (would mean the
    # README row is self-consistent and this repro is stale).
    return 0 if all_rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
