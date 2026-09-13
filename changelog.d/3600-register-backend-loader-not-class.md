### Fixed: `register_backend` refuses the backend class where a loader belongs

`register_backend(name, loader)` takes a zero-arg callable that *returns* the
backend class. A class is itself callable, so passing one satisfied the call the
registry makes and produced an *instance* where a class was expected: the
registration was accepted, and the failure surfaced later inside
`create_simulation()` as `AttributeError: 'MyEngine' object has no attribute
'__name__'` - a report naming neither the parameter nor the registration that
supplied it. `force=True` waived nothing here, because the mistake is not a name
conflict.

A `SimEngine` subclass passed as *loader* is now refused at the door with the
remedy (`lambda: MyEngine`, which also keeps the backend's import deferred). A
class-shaped factory that is *not* a `SimEngine` stays accepted - it is a
legitimate loader.

`docs/api-reference.md` spelled the parameter `cls`, so the documented call was
the mistake: the keyword form raised `TypeError: unexpected keyword argument
'cls'` and the positional form registered a class. The row now names `loader`,
and the reference's signature grader reads every code span on a table line
rather than only the leading one - the drifted signature sat second in a cell
that opened with `list_backends()`.
