### Fixed: a fetch that lands without the declared model is reported, not counted as a download

`download_assets` / `download_robots` graded a clone route by whether the copy
succeeded, not by whether the tree it copied holds the model the registry entry
declares. Both clone routes fetch their source repository at HEAD, so an
upstream rename retires the declared name while the directory around it
survives - and the copy still reported `Downloaded: 1, Failed: 0` for a robot
`resolve_model_path` then cannot find, whose refusal names the download that
just ran as its remedy. Measured against `mujoco_menagerie` HEAD today, three
shipped entries are in that state (`so101`, `unitree_a1`, `trossen_wxai`);
`robot_descriptions` hides it because it pins a commit instead of following
HEAD.

The verdict is now `failed: fetched tree has no <model_xml>, the model this
robot's registry entry declares - it holds <the model files that arrived>`. The
question "is the declared model on disk?" has one owner
(`_declared_model_absent`) reached from all four fetch paths, in place of the
`robot_descriptions` route grading it and the two clone routes not. Nothing is
deleted: the fetched files are a usable tree, `add_robot` still reaches such a
robot through its `scene_xml`, and a user's own files live in that directory -
only the verdict changes.
