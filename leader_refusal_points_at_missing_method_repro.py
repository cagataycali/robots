"""Repro: Robot('*_leader', mode='real') refusal points at attach_teleop that
does not exist on the native Feetech driver.

The refusal in ``strands_robots/robot.py:194-201`` universally recommends:

    Robot('<follower>', mode='real', port=...).attach_teleop('<leader>', port=...)

But ``attach_teleop`` is provided by ``strands_robots.teleop_mixin.TeleopMixin``,
which the lerobot hardware Robot inherits and the MuJoCo simulation inherits,
but the native ``strands_robots.drivers.feetech.driver.FeetechDriver`` does NOT
inherit. So a caller who picked ``driver='strands'`` (documented in
``docs/start/first-real-arm.md`` as one of two options) applies the fix from
the refusal and gets an AttributeError.

Reproduces cleanly on the current tree (HEAD e526c6d2d):

    $ python leader_refusal_points_at_missing_method_repro.py
    [refusal text follows]
    ...
    apply the fix: AttributeError: 'FeetechDriver' object has no attribute 'attach_teleop'

Expected behaviour: either
  (a) the refusal names driver='lerobot' as the driver the fix requires, or
  (b) FeetechDriver gains ``attach_teleop`` (it already implements ``send_action``,
      which is what the teleop loop calls, so the mixin should compose cleanly).
"""

from strands_robots import Robot, Teleoperator


def demonstrate_refusal_text() -> str:
    try:
        Robot("so101_leader", mode="real", port="/dev/null")
    except ValueError as exc:
        return str(exc)
    raise SystemExit("Robot('so101_leader', ...) unexpectedly did not refuse.")


def apply_the_refusal_fix() -> None:
    # first-real-arm.md shows this as one of two documented drivers.
    follower = Robot("so101", mode="real", driver="strands", transport="twin")
    try:
        # exact call the refusal text prescribes:
        follower.attach_teleop(Teleoperator("so101_leader", port="/dev/null", mock=True))
    except AttributeError as exc:
        print(f"apply the fix: AttributeError: {exc}")
    finally:
        follower.cleanup()


if __name__ == "__main__":
    print("=== refusal text ===")
    print(demonstrate_refusal_text())
    print()
    print("=== applying the recommended fix on the native driver ===")
    apply_the_refusal_fix()
