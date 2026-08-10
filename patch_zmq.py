import ast
import pathlib

P = pathlib.Path("tests/test_zmq_timeout_ms_domain.py")
src = P.read_text(encoding="utf-8")
orig = src

# ---------------------------------------------------------------- 1. stale label
LABEL_OLD = '    ("one-millisecond", 2, 2),\n'
LABEL_NEW = '    ("two-milliseconds", 2, 2),\n'
assert src.count(LABEL_OLD) == 1, src.count(LABEL_OLD)

# --------------------------------------------------- 2. derived round-trip table
CLIENTS_ANCHOR = "CLIENTS = [Gr00tInferenceClient, MoveIt2InferenceClient]\n"
assert src.count(CLIENTS_ANCHOR) == 1

CONSTANTS = '''#: Smallest budget a live round trip is asserted inside.
#:
#: A fresh REQ socket pays the TCP connect and the ZMQ handshake on its first
#: call, and that cost is scheduler-bound rather than transport-bound: measured
#: over a loopback sidecar on an idle host it is p50 0.24 ms / max 0.51 ms, and
#: under CPU contention it crosses 2 ms. So an answer required to arrive inside
#: a 2 ms budget asserts the host's scheduler, not the budget reaching the
#: socket - which is the property this file exists to pin, and which
#: ``getsockopt`` states exactly and without a clock. Budgets at or above this
#: floor keep three orders of magnitude of headroom over the connect cost, so
#: their round trip is a statement about the transport again.
MIN_ROUND_TRIP_BUDGET_MS = 1000

#: The usable budgets a live round trip is asserted inside.
#:
#: Derived from :data:`USABLE` rather than written out, so a budget added there
#: cannot acquire a wall-clock assertion by also being added to a second list,
#: and a tight one cannot be given one by hand.
ROUND_TRIP: list[tuple[str, Any, int]] = [row for row in USABLE if row[2] >= MIN_ROUND_TRIP_BUDGET_MS]

'''

# ------------------------------------------------- 3. split the wall-clock test
tree = ast.parse(src)
target = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "test_a_usable_budget_is_stored_as_an_int_and_still_pings":
        target = node
assert target is not None
first_line = min([d.lineno for d in target.decorator_list] + [target.lineno])
lines = src.splitlines(keepends=True)
old_block = "".join(lines[first_line - 1 : target.end_lineno])
assert "assert client.ping() is True" in old_block
assert "getsockopt(zmq.RCVTIMEO)" in old_block

SPLIT = '''    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize(("label", "value", "expected"), USABLE, ids=[c[0] for c in USABLE])
    def test_a_usable_budget_is_stored_as_an_int_and_reaches_both_socket_options(
        self, cls: type, label: str, value: Any, expected: int
    ) -> None:
        """Every usable spelling, asserted without a clock.

        ``getsockopt`` states the whole property this class exists to pin - that
        the coerced value is what the transport was configured with - and states
        it for the smallest usable budget as exactly as for the default one.
        """
        with sidecar_for(cls) as sidecar:
            client = cls(host="127.0.0.1", port=sidecar.port, timeout_ms=value)
            try:
                assert client.timeout_ms == expected
                assert type(client.timeout_ms) is int
                assert client.socket.getsockopt(zmq.RCVTIMEO) == expected
                assert client.socket.getsockopt(zmq.SNDTIMEO) == expected
            finally:
                client.socket.close()
                client.context.term()

    @pytest.mark.parametrize("cls", CLIENTS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize(("label", "value", "expected"), ROUND_TRIP, ids=[c[0] for c in ROUND_TRIP])
    def test_a_usable_budget_with_round_trip_headroom_still_answers(
        self, cls: type, label: str, value: Any, expected: int
    ) -> None:
        """The coerced budget still reaches a live sidecar.

        Parametrised over :data:`ROUND_TRIP` rather than :data:`USABLE`: the
        answer has to arrive inside the budget under test, so a budget without
        headroom over the connect cost would assert the host's scheduler here
        rather than the transport. See :data:`MIN_ROUND_TRIP_BUDGET_MS`.
        """
        with sidecar_for(cls) as sidecar:
            client = cls(host="127.0.0.1", port=sidecar.port, timeout_ms=value)
            try:
                assert client.ping() is True
            finally:
                client.socket.close()
                client.context.term()
'''

# ------------------------------------------------------------ 4. structural guard
GUARD_ANCHOR = "@requires_wire\nclass TestZmqStillTreatsTheseValuesAsMeasured:\n"
assert src.count(GUARD_ANCHOR) == 1

GUARD = '''class TestNoRoundTripIsAssertedInsideAScheduleBoundBudget:
    """Structural: a live answer is only required where the budget has room.

    The assertion this replaces read ``assert client.ping() is True`` inside a
    2 ms budget, so it held on an idle host and failed under CPU contention -
    a property of the scheduler rather than of the value reaching the socket.
    Checked over this module's own source rather than by naming today's tests,
    so the shape cannot return under a different name.
    """

    @staticmethod
    def _source() -> str:
        """This module's own source, read through its module object."""
        return pathlib.Path(inspect.getfile(inspect.getmodule(TestTheSharedDomain))).read_text()

    @staticmethod
    def _asserts_a_live_answer(fn: ast.AST) -> bool:
        """Whether this subtree contains ``assert <x>.ping() is True``."""
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare) or not node.ops:
                continue
            if not isinstance(node.ops[0], ast.Is):
                continue
            right = node.comparators[0]
            if not (isinstance(right, ast.Constant) and right.value is True):
                continue
            left = node.left
            if isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute) and left.func.attr == "ping":
                return True
        return False

    @staticmethod
    def _parametrize_tables(fn: ast.FunctionDef) -> set[str]:
        """Names of the module-level tables this test is parametrised over."""
        tables: set[str] = set()
        for dec in fn.decorator_list:
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                continue
            if dec.func.attr != "parametrize" or len(dec.args) < 2:
                continue
            values = dec.args[1]
            if isinstance(values, ast.Name):
                tables.add(values.id)
        return tables

    @classmethod
    def _tests_requiring_a_live_answer(cls, source: str) -> dict[str, set[str]]:
        """Map ``test name -> parametrised tables`` for every such test."""
        found: dict[str, set[str]] = {}
        for fn in ast.walk(ast.parse(source)):
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test_"):
                continue
            if cls._asserts_a_live_answer(fn):
                found[fn.name] = cls._parametrize_tables(fn)
        return found

    def test_the_scan_finds_every_test_requiring_a_live_answer(self) -> None:
        """Non-vacuity: a scan that found nothing would pass everything below."""
        assert set(self._tests_requiring_a_live_answer(self._source())) == {
            "test_the_default_budget_reaches_a_live_sidecar",
            "test_a_usable_budget_with_round_trip_headroom_still_answers",
        }

    def test_no_live_answer_is_required_inside_a_budget_without_headroom(self) -> None:
        offenders = {
            name: sorted(tables)
            for name, tables in self._tests_requiring_a_live_answer(self._source()).items()
            if "USABLE" in tables
        }
        assert not offenders, f"these require an answer inside a budget with no headroom: {offenders}"

    def test_the_scanner_detects_a_live_answer_over_the_full_table(self) -> None:
        """The scanner is answering the question, not returning ``{}``."""
        planted = (
            "@pytest.mark.parametrize(('label', 'value', 'expected'), USABLE)\\n"
            "def test_planted(label, value, expected):\\n"
            "    client = build(timeout_ms=value)\\n"
            "    assert client.ping() is True\\n"
        )
        assert self._tests_requiring_a_live_answer(planted) == {"test_planted": {"USABLE"}}

        no_round_trip = "def test_planted():\\n    assert client.timeout_ms == 2\\n"
        assert self._tests_requiring_a_live_answer(no_round_trip) == {}

    def test_the_round_trip_table_is_the_headroom_subset_of_usable(self) -> None:
        """Derived, so the two cannot drift apart."""
        assert ROUND_TRIP == [row for row in USABLE if row[2] >= MIN_ROUND_TRIP_BUDGET_MS]

    def test_the_floor_excludes_a_budget_and_keeps_one(self) -> None:
        """Non-vacuity for the derivation: the floor is doing work.

        A floor that excluded nothing would leave the wall-clock assertion in
        place; one that excluded everything would delete the round trip instead
        of moving it.
        """
        assert ROUND_TRIP
        assert len(ROUND_TRIP) < len(USABLE)
        assert [row[2] for row in USABLE if row not in ROUND_TRIP] == [2]

    def test_the_excluded_budget_keeps_every_assertion_needing_no_clock(self) -> None:
        """No coverage was traded away: it is still checked over the full table."""
        socket_option_test = next(
            fn
            for fn in ast.walk(ast.parse(self._source()))
            if isinstance(fn, ast.FunctionDef)
            and fn.name == "test_a_usable_budget_is_stored_as_an_int_and_reaches_both_socket_options"
        )
        assert "USABLE" in self._parametrize_tables(socket_option_test)


'''

# ------------------------------------------------------- 5. module docstring list
DOC_OLD = """* :class:`TestABudgetTheSiblingTransportsAcceptIsUsableHere` - the coercion,
  which is what makes this a fix rather than only a refusal.
"""
DOC_NEW = """* :class:`TestABudgetTheSiblingTransportsAcceptIsUsableHere` - the coercion,
  which is what makes this a fix rather than only a refusal.
* :class:`TestNoRoundTripIsAssertedInsideAScheduleBoundBudget` - structural, so
  a live answer is never required inside a budget the host's scheduler can
  exceed on a loaded runner.
"""
assert src.count(DOC_OLD) == 1

# ------------------------------------------------------------------ apply, in order
out = src
out = out.replace(LABEL_OLD, LABEL_NEW, 1)
out = out.replace(CLIENTS_ANCHOR, CONSTANTS + CLIENTS_ANCHOR, 1)
out = out.replace(old_block, SPLIT, 1)
out = out.replace(GUARD_ANCHOR, GUARD + GUARD_ANCHOR, 1)
out = out.replace(DOC_OLD, DOC_NEW, 1)

# ------------------------------------------------------------- post-conditions
assert out != orig
assert "one-millisecond" not in out
assert out.count("MIN_ROUND_TRIP_BUDGET_MS") == 4, out.count("MIN_ROUND_TRIP_BUDGET_MS")
assert out.count("test_a_usable_budget_is_stored_as_an_int_and_still_pings") == 0
tree2 = ast.parse(out)
# the module still parses, and the derived table lives above its first use
names = {n.name for n in ast.walk(tree2) if isinstance(n, ast.ClassDef)}
assert "TestNoRoundTripIsAssertedInsideAScheduleBoundBudget" in names
fns = {n.name for n in ast.walk(tree2) if isinstance(n, ast.FunctionDef)}
for expected in (
    "test_a_usable_budget_is_stored_as_an_int_and_reaches_both_socket_options",
    "test_a_usable_budget_with_round_trip_headroom_still_answers",
):
    assert expected in fns, expected
# exactly one live-answer assertion remains parametrised, and not over USABLE
lines_out = out.splitlines()
i_const = next(i for i, l in enumerate(lines_out) if l.startswith("ROUND_TRIP:"))
i_use = next(i for i, l in enumerate(lines_out) if "ids=[c[0] for c in ROUND_TRIP]" in l)
assert i_const < i_use, (i_const, i_use)
print(f"OK: {len(orig)} -> {len(out)} chars; ROUND_TRIP defined line {i_const + 1}, used line {i_use + 1}")
P.write_text(out, encoding="utf-8")
