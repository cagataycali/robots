### Fixed: a native driver no longer advertises a robot the factory cannot build

`FeetechDriver.SUPPORTED_ROBOTS` named `moss`. That name appeared in exactly one place in
the package -- the tuple itself. No entry in `registry/robots.json`, no lerobot robot type,
no asset. Registration and resolution are two halves of one chain: the seam maps a canonical
name to a driver class, and the factory maps that same name to a registry entry before it
builds anything. A name in the first and absent from the second registered cleanly, reported
itself through `list_native_drivers()`, and then raised
`ValueError: Unknown robot 'moss'` from `Robot("moss", mode="real", driver="strands")` -- the
exact refusal a native driver exists to remove, and the one
`TestAskingForANativeDriverThatIsNotThere` grades from the other side. `Robot("so101", ...)`
built a `FeetechDriver` on the same tree, so the failure was specific to the unregistered
name rather than to the driver.

The name is dropped rather than registered, because the comment directly above the tuple
already states the invariant -- "Every entry corresponds to a canonical name in
`strands_robots/registry/robots.json`" -- and its next sentence gives the disposition:
"registering for a robot we cannot verify is a promise this driver does not yet keep." A
robot with no registry entry, no lerobot type and no asset is exactly one we cannot verify.
The changelog fragment that introduced the driver described the same set as "every
Feetech-servo robot in the registry, and no other", which was true of the other five names;
this corrects that record.

Nothing in the tree related the two halves. `tests/test_driver_seam.py`, which grades every
shipped driver against the seam, contained no registry lookup at all. The feetech cell named
for the failure, `test_every_supported_robot_registers_the_driver_on_package_import`, asserts
that each advertised name resolves to `FeetechDriver` -- the seam half -- while its own
docstring names the consequence that lives in the other half ("a missing entry surfaces as
`Robot(...)` raising `ValueError`"). `moss` satisfied that cell.

So the relation is now graded twice. `tests/test_driver_seam.py` derives it over every entry
in `_SHIPPED_DRIVERS`, through the same `shipped_robot_names` helper the registration itself
uses, so a sixth driver is held to it the hour it lands instead of inheriting an exemption by
being absent from a list; a non-vacuity cell pins that the derivation reaches all five
shipped drivers, and a third cell registers a complete driver for an unregistered name and
drives the factory refusal that makes the relation worth having. `tests/drivers/
test_feetech_driver.py` keeps the local half beside the tuple, because that is where a name
gets added. Reverting the source change fails both, one in each file; deleting either cell
leaves the other catching it; deleting both is silent, which is the state that shipped.
