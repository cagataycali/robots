### Fixed: a body's reported mesh is its visual asset, not its collision hull

MJCF bodies routinely carry two mesh geoms: the visual asset a renderer should show, and a convex
hull the solver should collide. `_find_body_mesh` picks the one a reported `SceneObject` carries,
and it picked "the first geom whose `group` is not `"0"`" - reading a non-default MuJoCo group as
the visual marking.

That is backwards for the dominant convention. MuJoCo Menagerie marks a visual geom with
`contype="0" conaffinity="0"` and a collision geom with `group="3"`, and the collision geom is
routinely declared first, so the rule returned the hull. In the shipped `shadow_dexee` hand the
visual class carries no `group` at all - `<default class="finger/visual"><geom contype="0"
conaffinity="0"/>` - so each finger skipped its eight visual mesh geoms for the ninth,
`r3_finger_base_col`. `load_mjcf_scene_objects` reported those fingers as `r3_finger_base_col.stl`
under a successful load while `hand_base` in the same file reported its visual STL: one reader,
one file, two answers. `group` also cannot be the signal because the conventions built on it
disagree - robosuite emits visual geoms as `group="1"` where Menagerie spells collision as
`group="3"` - and MuJoCo attaches no meaning to it at all, it being a visualiser toggle.

The candidates are now ranked, strongest first, rather than the first non-default group winning.
MuJoCo lets two geoms touch only when `contype1 & conaffinity2` or `contype2 & conaffinity1` is
non-zero, so a geom declaring both as `0` cannot collide with anything and exists purely to be
looked at; that is the format's own statement of intent and it ranks first. A non-default group
stays as the weaker hint, taken when the contact declaration says nothing and never over a
contact-free geom. Document order still breaks a tie, so a subtree carrying no visual marking at
all reports the same mesh it always did. A stronger candidate in a nested body wins over a weaker
one on the body itself, which is the shape `shadow_dexee` has.

Graded across the downloaded asset corpus - 554 MJCF files, 2736 bodies declaring a mesh geom -
165 bodies in 18 files change their reported mesh and every one of them changes to a geom MuJoCo
cannot collide with: 162 from a group-marked hull, 2 from an unmarked geom, and one tie between
two contact-free geoms. No body moves the other way, no reported position or orientation changes
for a body whose mesh is unchanged, and neither reading raises. The files include the canonical
Menagerie assets `shadow_dexee` (19 bodies), `franka_fr3`, `franka_fr3_v2`,
`boston_dynamics_spot/spot_arm` and `pal_tiago_dual`.

Deliberately unchanged: a subtree where no geom is contact-free. `google_barkour_v0` has one set
of meshes serving both roles, so its `group="1"` marking is the only signal in the file and it
still decides; dropping the group tier instead of demoting it would have cost those bodies their
answer, and that is pinned as a control. The rank is internal to the pick - the one production
caller discards it - so nothing downstream of the reader gains a field.
