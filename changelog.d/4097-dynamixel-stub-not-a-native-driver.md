### Removed: the Dynamixel driver stub, so `driver="strands"` refuses a Dynamixel robot instead of building one that cannot move

`DynamixelDriver` opened no serial port and refused every write, yet it was
registered for koch, aloha, vx300s, wx250s, trossen_wxai and dynamixel_2r, so
`Robot("koch", mode="real", driver="strands")` returned a robot under success
that could neither read nor move a joint, and `list_driver_coverage()` reported
`strands` for all six. The stub and its registration are removed; the Protocol
2.0 codec (`strands_robots.drivers.dynamixel`) stays. Replacement: koch moves
through lerobot (`driver="lerobot"`, `pip install 'lerobot[dynamixel]'`); the
other five have no driver of either kind until a native bus lands, and the
`driver="strands"` refusal now says so instead of pointing at lerobot.
