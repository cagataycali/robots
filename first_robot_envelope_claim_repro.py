"""Repro for `docs/start/first-robot.md` line 78 mismatch.

The paragraph after the 'What each call did' table claims:

    "Every method returns the same envelope: `status` and a `content` list
     of `text`, `json` or `image` blocks."

That table names six methods (send_action, get_robot_state, step,
get_observation, render, cleanup). Only four of the six actually return
that envelope. Two do not:

  * `cleanup()`   -> None    (signed `-> None` in the base + every backend)
  * `get_observation()` -> a raw obs dict without `status`/`content`
                           (schema documented in
                            strands_robots/simulation/base.py:1649-1680:
                            "<joint>", "<joint>.vel", "<camera_name>")

A reader who trusted the paragraph writes
`robot.cleanup()["content"][0]["text"]` and gets a `TypeError`.
The table row for `get_observation` already self-describes it as "the
flat observation a policy sees" - so the mismatch is really the
paragraph over-generalising past what the table itself says.

Run: python first_robot_envelope_claim_repro.py
"""

from strands_robots import Robot


def main() -> None:
    robot = Robot("so101")
    try:
        obs = robot.get_observation()
        assert "status" not in obs, (
            "get_observation() has no `status` key - "
            "raw dict per base.py:1649 schema, not the envelope."
        )
        print(
            "get_observation() -> "
            f"type={type(obs).__name__} has_status={'status' in obs}"
        )

        ret = robot.cleanup()
        assert ret is None, (
            "cleanup() is signed `-> None` on every SimEngine backend "
            "(base.py:6854, mujoco:8313, newton:2800, isaac:9949). "
            "It cannot be indexed as an envelope."
        )
        print(f"cleanup() -> type={type(ret).__name__} value={ret!r}")

        # This is what the docs paragraph tells the reader to do:
        try:
            robot.cleanup()["content"][0]["text"]  # noqa: B018
        except TypeError as e:
            print(f"docs-following pattern raises: TypeError: {e}")
    finally:
        # `cleanup()` on a cleaned engine is a no-op; safe to call twice.
        try:
            robot.cleanup()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
