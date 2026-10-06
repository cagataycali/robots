### Fixed: an unknown teleop layout name points at the one you meant

`teleop_layout("g1_joint_28")` used to refuse with only the list of layouts.
The refusal now names the closest layouts first, compared case- and
dash-insensitively, so `G1_JOINT_29` and `blog-31-66` are answered with
`'g1_joint_29'` and `'blog_31_66'`. A name close to nothing still gets the list
alone.
