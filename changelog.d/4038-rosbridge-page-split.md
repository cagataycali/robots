### Docs: the rosbridge page keeps the tool, driving a robot over it gets its own page

`docs/rosbridge-integration.md` documented two subjects in one 1,592-word scroll:
the `use_rosbridge` tool (install, the actions table, examples, the security
posture) and how a mobile base drives over that transport - `RosbridgeRobot`'s
constructor, its address and drive contracts, `from_curiosity`, and the NASA
Curiosity Docker recipe. The second half moves to `docs/ros2/rosbridge-robot.md`,
leaving 504 and 1,165 words, so neither page owes the 1,500-word budget an
exemption. The moved text is unchanged apart from the two relative links that
follow it into `docs/ros2/`; the tool page keeps a pointer under the heading it
used to own. The drive-contract grader and the Curiosity example's docstring
follow the section to the page it now lives on, and the new page has a nav row.
