### Fixed: a camera removal the scene will not recompile without is refused, not reported as done

``remove_camera`` deletes the camera element from the live ``MjSpec`` before the
``spec.recompile`` that validates the result, and a refused recompile leaves the
compiled model untouched. It logged that refusal at warning level, dropped the
registry entry anyway and returned ``{"status": "success", ... "removed."}``, so
the spec stopped declaring a camera the model still had: ``list_cameras`` no
longer named it while ``render`` and ``get_camera_params`` went on resolving it,
and the delete landed later, applied by whichever unrelated scene mutation next
recompiled successfully -- a camera a rollout or recording was reading from
disappearing at an ``add_object`` call with nothing to attribute it to.

The delete now takes a spec snapshot first and is rolled back out on a refusal,
which is reported as an error naming the camera and saying it is still
registered. That is the contract its inverse already had: ``add_camera`` rolls a
refused add back out and reports it. The registry entry is dropped only once the
recompile is accepted, so a refused removal leaves the scene -- registry, spec
and model -- exactly as it was found, and the identical call succeeds once
whatever made the scene uncompilable clears.

The removal also routes through the shared ``_recompile_preserving_state`` path
every other scene mutation uses instead of its own ``spec.recompile`` call, so it
picks up the forward pass that populates camera transforms and the cached-XML
sync that logs a failed ``spec.to_xml()`` rather than discarding it silently.
