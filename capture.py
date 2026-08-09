"""Measure what reaches the lerobot argv for each play_sounds spelling."""
import contextlib, importlib, io, json, pathlib, sys

import strands_robots.tools.lerobot_teleoperate as m

TREE = str(pathlib.Path(m.__file__).parents[2])
print("TREE:", TREE)
B = m.build_lerobot_command
COMMON = dict(
    robot_type="so101_follower", robot_port="/dev/ttyACM1",
    teleop_type="so101_leader", teleop_port="/dev/ttyACM0",
)
MODES = {
    "record":  dict(action="start",  dataset_repo_id="user/pick", dataset_single_task="pick the cube", **COMMON),
    "replay":  dict(action="replay", dataset_repo_id="user/pick", **COMMON),
    "dagger":  dict(action="dagger", dataset_repo_id="user/pick", policy_path="lerobot/act_so101", **COMMON),
    "teleoperate": dict(action="start", **COMMON),
}
CONFIGS = {"record": ("lerobot.scripts.lerobot_record", "RecordConfig"),
           "replay": ("lerobot.scripts.lerobot_replay", "ReplayConfig")}


def token(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def verdict(mode, value):
    """What the builder does with this spelling: emitted value / refusal / accept."""
    try:
        argv = B(play_sounds=value, **MODES[mode])
    except ValueError as exc:
        return ("refused", "play_sounds" in str(exc))
    tok = token(argv, "--play_sounds")
    return ("emitted", tok) if tok is not None else ("accepted-inert", None)


def round_trip(mode, want):
    """What lerobot's own CLI parses out of the argv we emit."""
    if mode not in CONFIGS:
        return None
    module, name = CONFIGS[mode]
    try:
        importlib.import_module("lerobot.policies")
        config = getattr(importlib.import_module(module), name)
        argv = B(play_sounds=want, **MODES[mode])[3:]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            parsed = __import__("draccus").parse(config_class=config, args=argv)
        return bool(getattr(parsed, "play_sounds"))
    except BaseException as exc:  # noqa: BLE001 - any failure is the measurement
        return f"error:{type(exc).__name__}"


facts = {"tree": TREE, "modes": {}, "grid": {}, "roundtrip": {}}
for mode in MODES:
    on, off = B(play_sounds=True, **MODES[mode]), B(play_sounds=False, **MODES[mode])
    facts["modes"][mode] = {
        "true_token": token(on, "--play_sounds"),
        "false_token": token(off, "--play_sounds"),
        "identical_argv": on == off,
        "argv_off": off,
    }
    for want in (True, False):
        facts["roundtrip"][f"{mode}:{want}"] = round_trip(mode, want)

SPELLINGS = ["true(bool)", "false(bool)", '"false"', '"off"', "None", "[]", "0", "1"]
VALUES = {"true(bool)": True, "false(bool)": False, '"false"': "false", '"off"': "off",
          "None": None, "[]": [], "0": 0, "1": 1}
for mode in MODES:
    facts["grid"][mode] = {s: verdict(mode, VALUES[s]) for s in SPELLINGS}

out = pathlib.Path(sys.argv[1])
out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
print("wrote", out)
