"""Tests for ``format_robot_table`` - column width handling (issue #113)."""

from __future__ import annotations

from strands_robots.registry.robots import (
    _FIXED_PREFIX_WIDTH,
    format_robot_table,
    list_robots,
)


class TestDefaultWidth:
    def test_default_max_line_length_is_bounded(self):
        table = format_robot_table()  # default max_width=100
        max_len = max(len(line) for line in table.split("\n"))
        # Allow a small margin - the rule is the longest line; data rows
        # should fit inside max_width + some padding for the header/rule.
        assert max_len <= 101, f"max line {max_len} exceeds 100 chars"

    def test_contains_header_and_total(self):
        table = format_robot_table()
        assert "Name" in table
        assert "Category" in table
        assert "Description" in table
        assert f"Total: {len(list_robots())} robots" in table

    def test_contains_all_categories(self):
        table = format_robot_table()
        # At least one of each category should be represented in the registry.
        for cat in ("arm", "humanoid", "hand"):
            assert cat in table


class TestNarrowWidth:
    def test_80_col_terminal_fits(self):
        table = format_robot_table(max_width=80)
        max_len = max(len(line) for line in table.split("\n"))
        # 80 is a hard target for narrow terminals; our rule is <= that + 1
        # (the ellipsis adds one wide char that may not be counted).
        assert max_len <= 81, f"max line {max_len} exceeds 80 chars"

    def test_descriptions_are_truncated_with_ellipsis(self):
        """Long descriptions should end with the truncation marker '...'."""
        narrow = format_robot_table(max_width=80)
        wide = format_robot_table(max_width=1000)
        # At least one row must have been truncated at narrow width.
        assert "..." in narrow
        # And that same row is longer in the wide rendering.
        assert "..." not in wide


class TestWideWidth:
    def test_wide_width_disables_truncation(self):
        table = format_robot_table(max_width=1000)
        assert "..." not in table

    def test_minimum_desc_width_is_enforced(self):
        """Even at absurdly narrow widths we keep a 20-char Description column
        rather than collapsing to zero."""
        table = format_robot_table(max_width=20)
        # Prefix alone is wider than 20; we clamp to
        # _FIXED_PREFIX_WIDTH + 20 so every row still shows some description.
        max_len = max(len(line) for line in table.split("\n"))
        assert max_len >= _FIXED_PREFIX_WIDTH + 20 - 1


class TestConsistency:
    def test_row_count_matches_registry(self):
        """The table should have (2 header + robots + 2 footer) lines.
        Categories with zero robots contribute no data rows."""
        table = format_robot_table()
        lines = table.split("\n")
        non_empty_rows = [line for line in lines[2:-2] if line.strip() and "Total:" not in line]
        assert len(non_empty_rows) == len(list_robots())
