### Docs: every row of the install page's extras table says what the extra is for

`[flux3]`, `[holosoma]`, `[xarm]`, `[rby1]`, `[spot]` and `[voice]` rendered
with an empty purpose column, so they read like placeholders. Each now has one
clause, and a test fails when an extra in `pyproject.toml` has no purpose or a
purpose names an extra that no longer exists.
