### Changed: finalizer tests release the object instead of calling `__del__`

Three test cells invoked another object's `__del__` as an ordinary method. That
grades the method body while saying nothing about the finalization a caller
gets: the object is still referenced, so a guard that depends on it being
collected -- or a teardown whose cost is the point, like a socket that must not
block on close -- runs in a state the interpreter never produces. They now drop
the last reference and collect, which is the idiom
`tests/test_hardware_cleanup_survives_a_failed_robot_init.py` already uses and
pins the reasoning for. A new sweep,
`tests/test_finalizer_cells_are_driven_by_the_interpreter.py`, keeps the call
out of the test areas: CodeQL's `py/explicit-call-to-delete` fires on it and
`required_review_thread_resolution` turns that alert into a merge gate, so the
idiom blocked whoever wrote the next finalizer test.
