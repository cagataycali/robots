"""No React hook may be declared after a component's early return (BUGS.md Q77).

Q77: SettingsDrawer grew a `useState` BELOW its `if (!open) return null`. A closed drawer therefore
ran one hook fewer than an open one, and React refuses that:

    Rendered more hooks than during the previous render.

Effect: the settings screen was DEAD -- every open hit the error boundary ("settings stopped working"),
and because the boundary stays crashed, the audit then reported help, robot detail and train as broken
too. It shipped because nothing in this repo checks the Rules of Hooks: eslint-plugin-react-hooks is not
wired into a run anybody makes, the pure .test.mjs files test lib/ (no hooks), and the python suite never
looked at .tsx. A rule with no runner is a comment.

So this is a static scan in the suite everyone runs: for each component file, find the first
component-level early return and refuse any hook call after it. It is deliberately dumb (regex, one
indentation level, generics included -- `useMemo<Drafts>(` is why the first hand-grep missed Q77) and it
reports the file and line rather than pretending to be a type checker.
"""

from __future__ import annotations

import re
from pathlib import Path

COMPONENTS = Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard" / "frontend" / "src"

#: `useThing(` or `useThing<T>(` -- the generic form is the one a hand-written grep loses.
HOOK = re.compile(r"\buse[A-Z][A-Za-z0-9]*\s*(?:<[^;]*?>)?\s*\(")
#: a return at component-body indentation: `  return …` or `  if (…) return …`
EARLY_RETURN = re.compile(r"^ {2}(?:if\s*\(.*\)\s*)?return\b")
#: where a component body starts, so returns inside the small helpers above it are not the anchor
COMPONENT_START = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:export\s+default\s+)?function\s+[A-Z]")


def _hooks_after_early_return(source: str) -> list[tuple[int, str]]:
    """Every hook call that a component reaches only on some renders."""
    lines = source.split("\n")
    body_started = False
    early: int | None = None
    offenders: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        # Any top-level declaration ENDS the previous component's hook region. Without this, a
        # module-level one-liner like `export const useConfig = () => useContext(CTX)` inherited the
        # region of the component above it and was reported as a late hook -- a false positive, and a
        # guard that cries wolf is a guard somebody deletes.
        if re.match(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|const|let|class)\b", line):
            is_component = bool(
                COMPONENT_START.match(line) or re.match(r"^(?:export\s+)?const\s+[A-Z]\w*\s*[:=].*=>", line)
            )
            body_started, early = is_component, None
            continue
        if not body_started:
            continue
        if early is None and EARLY_RETURN.match(line):
            early = i
            continue
        if early is not None and HOOK.search(line) and not line.lstrip().startswith(("//", "*", "import")):
            offenders.append((i, line.strip()))
    return offenders


def test_no_hook_is_declared_after_an_early_return() -> None:
    problems: list[str] = []
    for path in sorted(COMPONENTS.rglob("*.tsx")):
        for line_no, text in _hooks_after_early_return(path.read_text()):
            problems.append(f"{path.relative_to(COMPONENTS)}:{line_no}: {text[:90]}")
    assert not problems, (
        "a hook here runs on some renders and not others, which React ends with "
        '"Rendered more hooks than during the previous render" and the error boundary turns into a '
        "dead screen (Q77). Move the declaration above the early return:\n  " + "\n  ".join(problems)
    )


def test_the_scanner_catches_the_shape_that_killed_the_settings_screen() -> None:
    """Non-vacuity, written as Q77's actual code: state below `if (!open) return null`."""
    offenders = _hooks_after_early_return(
        "export default function SettingsDrawer({ open }: { open: boolean }) {\n"
        "  const [query, setQuery] = useState('')\n"
        "  if (!open) return null\n"
        "  const [connVerdict, setConnVerdict] = useState<ConnectionVerdict | null>(null)\n"
        "  return <div />\n"
        "}\n"
    )
    assert [n for n, _ in offenders] == [4], offenders
    assert "connVerdict" in offenders[0][1]


def test_a_hook_before_the_early_return_is_fine() -> None:
    """The legal shape must not be flagged, or the guard gets switched off."""
    assert not _hooks_after_early_return(
        "export default function Sheet({ open }: { open: boolean }) {\n"
        "  const [q, setQ] = useState('')\n"
        "  const rows = useMemo<Row[]>(() => [], [])\n"
        "  if (!open) return null\n"
        "  return <div>{q}{rows.length}</div>\n"
        "}\n"
    )


def test_a_module_level_custom_hook_after_a_component_is_not_an_offender() -> None:
    """The false positive this scanner had on its first run, pinned so it cannot come back."""
    assert not _hooks_after_early_return(
        "export function ConfigProvider({ children }: { children: ReactNode }) {\n"
        "  const [config, setConfig] = useState(null)\n"
        "  return <CTX.Provider value={config}>{children}</CTX.Provider>\n"
        "}\n"
        "export const useConfig = () => useContext(CTX)\n"
    )


def test_a_helper_functions_return_is_not_the_anchor() -> None:
    """Helpers above a component return early all the time; that says nothing about its hooks."""
    assert not _hooks_after_early_return(
        "function fmt(v: number | null) {\n"
        "  if (v === null) return '--'\n"
        "  return v.toFixed(2)\n"
        "}\n"
        "export default function Panel() {\n"
        "  const [x, setX] = useState(0)\n"
        "  return <div>{fmt(x)}</div>\n"
        "}\n"
    )
