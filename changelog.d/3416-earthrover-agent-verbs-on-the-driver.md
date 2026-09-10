### Changed: the rover's agent verbs moved onto its driver, where a model can reach them

``EarthRoverDriver.tool_spec`` now declares one verb per capability the SDK
exposes - ``sensors``, ``status``, ``camera``, ``move``, ``lamp``, ``speak``,
``stop`` - and ``stream`` dispatches each to the method that already owns its
judgement. ``Agent(tools=[rover])`` therefore drives the rover it can see.

The six ``rover_*`` ``@tool`` functions this replaces are removed. Each required
``driver: Any``, which reached the model's schema as a required parameter with no
type at all, so an agent could not call one: a handle it invented was refused
("``driver`` of type 'str' does not expose a twist write"), and omitting it
failed schema validation. Meanwhile the driver's own schema declared only reads
and a halt - a rover an agent could watch and not drive.

Every contract the verbs carried survives on the driver. A timed ``move`` holds
the twist for at most ``MAX_MOVE_DURATION_S`` and reports **both** halves, so a
lost trailing stop is an error saying the rover may still be rolling rather than
a completed move. ``lamp`` still rides a zero twist, because the SDK carries the
lamp inside the one ``/control`` frame. ``sensors`` summarises battery, signal,
heading and GPS in one line beside the full snapshot, and an empty cache is a
refusal naming the remedy instead of an empty success. ``camera`` still answers
with an image block the model can see, while ``capture_frame`` keeps the base64
JSON the mesh publishes.

Two behaviours are new. A coordinate the SDK reported as something other than a
number now reads as "no fix" instead of raising ``ValueError`` out of the
summary's format string and losing the battery reading beside it. And
``EarthRoverDriver.move`` takes ``duration_s``, so the bounded hold is available
to a Python caller and not only through an agent.

``tests/drivers/test_a_robot_verb_needs_no_handle_a_model_cannot_send.py``
records the rest of that population: a ``@tool`` requiring a live handle is now
an exact recorded set, so a new one fails instead of shipping.
