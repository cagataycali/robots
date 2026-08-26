### Tests: a `system_install=` remedy is graded by the string it carries, not the keyword's presence

`tests/test_dependency_audit.py::test_require_optional_offers_no_pip_remedy_for_a_system_provided_module`
exists because `rclpy` and `rosidl_runtime_py` are not published on PyPI, so every
`pip install` line offered for them is an instruction a caller can follow to no
effect: `pip install 'strands-robots[ros2]'` exits 0 having installed only the
cyclonedds RMW binding and leaving the module exactly as missing, and
`pip install rclpy` does not resolve at all. `require_optional(system_install=...)`
replaces that block with the step that does supply the module, and the sweep
required every such call site to pass it.

It required the keyword and not the string. Those are different claims, and the
gap is not narrow - measured against `require_optional` itself with the module
made unimportable, three literals satisfy the keyword's presence and defeat what
it is for:

* `system_install=None` renders the pip block **byte-for-byte identical** to
  omitting the keyword. It is the parameter's own default and inside its declared
  `str | None` annotation, so it is the natural spelling for a caller forwarding a
  per-platform hint that turned out to be absent - and it produces exactly the
  message this sweep was written to forbid.
* `system_install=""` (or any blank string) leaves the refusal naming the module
  and carrying no remedy at all.
* A non-string remedy replaces the documented failure: `str.join` raises
  `TypeError`, so a caller writing `except ImportError` around an optional import
  misses the refusal entirely.

Both production sites pass a module-level constant, so nothing in the tree is an
offender and this is a hole in the guard rather than a live defect. It is not
covered elsewhere either. `test_the_rclpy_refusals_name_the_step_that_supplies_it`
does assert on the real messages, which is why planting `system_install=None` at
the `rclpy` site is caught - but it names two `rclpy` call sites explicitly, and
`rosidl_runtime_py` is the other member of the set the sweep grades and has no
behavioural test at all. Planting `require_optional("rosidl_runtime_py",
system_install=None)` therefore fired nothing: the sweep reported the keyword as
present and the suite passed.

The rule now reads the value, through one predicate the sweep and the exemplars
share, and each reason names the message that spelling really renders rather than
only reporting that the value was rejected. Its boundary is deliberate: only a
literal is judged. A name, an attribute or an f-string is the shipped form, whose
text a syntax tree cannot know, so it is accepted here and left to the runtime
tests in `tests/test_utils.py`. Because the tree holds no offender the corpus
cannot exercise the new branch, so the exemplars grade the predicate directly -
four accepted forms, five refused ones, and a non-vacuity check that the rule
reaches both verdicts - with the consequence of each refused literal measured
against `require_optional` beside them.

`require_optional` is unchanged, and that is a measurement rather than an
omission. Reading `system_install` for truth instead of for `None` would send a
blank *and* a non-string remedy back to the pip block, which for these libraries
is the dead-end instruction; the current `is not None` branch is the safer of the
two, and the spellings that should never be written are now refused where they
are written rather than absorbed where they are read.
