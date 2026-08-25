### Fixed: a forced processor-module re-import resets every step it owns, not a subset

`tests/policies/lerobot_local/test_vla_jepa.py` is the regression guard for "the generic
`lerobot_local` path already carries a brand-new VLA": it forces a fresh import of
`lerobot.policies.vla_jepa.processor_vla_jepa` so the `@ProcessorStepRegistry.register`
decorators run again, then asserts `_register_policy_processor_steps` put VLA-JEPA's bespoke
postprocessor steps back. To force that import it evicted the module from `sys.modules` and
cleared the step names -- but it cleared three names where the module registers four, leaving
`vla_jepa_image_prep` behind.

LeRobot refuses a duplicate: `ProcessorStepRegistry.register` raises `ValueError: Processor step
'<name>' is already registered`. `vla_jepa_image_prep` is the module's FIRST decorator, so the
re-import died on its way in, the three later steps never registered, and the test failed with
`KeyError: Processor step 'vla_jepa_clip_actions' not found in registry` -- reporting the
registration path for a state the reset had created. Measured on lerobot 0.6.2: given a reset that
covers all four names, `_register_policy_processor_steps("vla_jepa")` registers all four, so the
path being graded was working the whole time.

The guard therefore had teeth in neither direction. Its own skip guard calls `list_policy_types()`,
which imports every policy config module and so imports the `vla_jepa` package, whose `__init__`
imports the processor module -- meaning the survivor is always registered by the time the body
runs. So the test fails wherever `vla_jepa` is registered, which is exactly where it is meant to
have teeth, and skips wherever it is not.

It also left the process worse than it found it. With the module absent from `sys.modules` and one
of its names still registered, every later import of that module trips the same surviving name, so
the module is un-importable for the remainder of the run: a probe that imports it after this file
executes raises `ValueError: Processor step 'vla_jepa_image_prep' is already registered`, and it
passes when run without this file. Any future test touching VLA-JEPA in the same session would
have failed for a reason belonging here.

The reset is now read off the module rather than listed -- every registry entry whose step class
has that `__module__` -- so a step lerobot adds, renames or moves is covered with no edit here,
which matters because a name the reset misses is a name the re-import cannot re-register. Registry
entries and the `sys.modules` entry are both restored on exit, and the original step classes are
put back rather than the new objects the fresh import built, so a reference another test already
holds and the registry entry stay the same class.

The assertions still concern the three postprocessor steps a checkpoint's
`policy_postprocessor.json` names; a non-vacuity check now requires those three to be among the
names the module owns, so an upstream rename fails here naming both sides instead of passing on an
empty intersection. Two tests are added: one pinning that the registry and `sys.modules` are left
as they were found, one pinning the surviving-name hazard as behaviour, so the reason the reset has
to be total is carried by a test rather than by a comment.

`test_sys_modules_removal_leaves_no_orphan` grades unrestored `sys.modules` removals, and it is
right not to have flagged this one: its rule covers a removal that orphans a reference some test
*patches*, and nothing patches this module. The harm here comes from the same unrestored removal by
a different route -- the module's import has a side effect that is not idempotent -- so the
restoration is carried in the file that does the removing.

No runtime code changes: `git diff --name-only -- strands_robots/` against the merge base is empty.
