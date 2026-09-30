### Fixed: `contact_between` fires for the object names a scene lists

`add_object("cube")` builds a geom named `cube_geom`, and `get_contacts` reports
that name. `contact_between` compared exact geom names, so
`contact_between(geom_a="cube", geom_b="tray")` stayed `False` while the two
objects were touching. Nothing warned about it, and a `success` or `stop_when`
clause written with the object names scored 0 percent. Each side now matches a
geom that is the name, or a geom that belongs to the body or object of that
name. This is the mapping `grasped` and `body_on(require_contact=True)` already
use. Exact geom names match as before.
