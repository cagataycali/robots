"""Measure the three candidate handler widths on the verdict classifier."""
from __future__ import annotations
import pathlib, sys
from typing import Any
import pytest
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])


# ---- the three candidate classifiers, byte-for-byte the same but for the handler ----
def classify_base(cls: type) -> str | None:
    factory: Any = cls
    obj = factory.__new__(factory)
    try:
        rendered = repr(obj)
    except BaseException as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    if cls.__name__ not in rendered:
        return f"does not identify its type: {rendered!r}"
    return None


def classify_none(cls: type) -> str | None:
    factory: Any = cls
    obj = factory.__new__(factory)
    rendered = repr(obj)
    if cls.__name__ not in rendered:
        return f"does not identify its type: {rendered!r}"
    return None


def classify_exception(cls: type) -> str | None:
    factory: Any = cls
    obj = factory.__new__(factory)
    try:
        rendered = repr(obj)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    if cls.__name__ not in rendered:
        return f"does not identify its type: {rendered!r}"
    return None


VARIANTS = [("except BaseException", classify_base), ("no handler", classify_none), ("except Exception", classify_exception)]


def _cls(name: str, body):
    return type(name, (), {"__repr__": body})


# ---- the nine rows: 5 library outcomes the survey exists to collect, 4 control-flow classes ----
def _raise(exc):
    def r(self):
        raise exc
    return r


ROWS = [
    # (label, kind, class)
    ("AttributeError (the real defect)", "library",
     _cls("Defect", lambda self: f"<Defect {self.node_name}>")),
    ("TypeError inside repr", "library", _cls("Typed", _raise(TypeError("bad format")))),
    ("RuntimeError inside repr", "library", _cls("Runt", _raise(RuntimeError("no context")))),
    ("repr hides its own type", "library", _cls("Anon", lambda self: "<object>")),
    ("repr is correct (survives)", "library", _cls("Good", lambda self: "<Good ok>")),
    ("KeyboardInterrupt", "control", _cls("KInt", _raise(KeyboardInterrupt()))),
    ("SystemExit", "control", _cls("SExit", _raise(SystemExit(2)))),
    ("pytest.skip", "control", _cls("Skip", _raise(pytest.skip.Exception("needs a dep")))),
    ("pytest.fail", "control", _cls("Fail", _raise(pytest.fail.Exception("explicit fail")))),
]

print("\n=== MRO check: the four control-flow classes ===")
for label, kind, cls in ROWS:
    if kind != "control":
        continue
    try:
        cls().__repr__()
    except BaseException as exc:
        print(f"  {label:20s} Exception-subclass={isinstance(exc, Exception)} qualname={type(exc).__qualname__}")

print("\n=== 3 variants x 9 rows ===")
header = f"{'row':36s} {'kind':8s} " + " ".join(f"{n:22s}" for n, _ in VARIANTS)
print(header)
print("-" * len(header))
score = {n: {"collected": 0, "escaped": 0} for n, _ in VARIANTS}
for label, kind, cls in ROWS:
    cells = []
    for name, fn in VARIANTS:
        try:
            verdict = fn(cls)
            if verdict is None:
                cells.append("None (survives)")
                if kind == "library" and label.startswith("repr is correct"):
                    score[name]["collected"] += 1
            else:
                cells.append(f"verdict {verdict[:14]!r}")
                if kind == "library":
                    score[name]["collected"] += 1
        except BaseException as exc:
            cells.append(f"ESCAPES {type(exc).__name__}")
            if kind == "control":
                score[name]["escaped"] += 1
    print(f"{label:36s} {kind:8s} " + " ".join(f"{c:22s}" for c in cells))

print("\n=== score ===")
for name, _ in VARIANTS:
    s = score[name]
    print(f"  {name:22s} library outcome collected {s['collected']}/5   control flow escapes {s['escaped']}/4")
